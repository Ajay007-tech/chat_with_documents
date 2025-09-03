import os
import pprint
import re
import torch
import gc
import psutil
from typing import List, Optional, Union

from qwen_agent import Agent, MultiAgentHub
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.gui.gradio_utils import format_cover_html
from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot, get_avatar_image
from qwen_agent.llm.schema import CONTENT, FILE, IMAGE, NAME, ROLE, USER, Message
from qwen_agent.log import logger
from qwen_agent.utils.utils import print_traceback

try:
    from patching import common_programming_language_extensions
except ImportError:
    common_programming_language_extensions = [
        "py", "java", "cpp", "c", "h", "cs", "js", "ts", "rb", "php",
        "swift", "go", "rs", "kt", "scala", "m", "css", "sql", "sh",
        "pl", "r", "jl", "dart", "json", "xml", "yml", "toml"
    ]

class WebUI:
    """A Common chatbot application for agent with lazy loading support."""

    def __init__(self, agent: Union[Agent, MultiAgentHub, List[Agent], None], 
                 chatbot_config: Optional[dict] = None, 
                 model_manager=None):
        """
        Initialization the chatbot.
        Args:
            agent: The agent or None for lazy loading
            chatbot_config: The chatbot configuration
            model_manager: Model manager for dynamic model loading
        """
        chatbot_config = chatbot_config or {}

        # Store model manager for dynamic loading
        self.model_manager = model_manager
        self.enable_quantization_selector = chatbot_config.get('quantization_selector', False)
        self.lazy_loading = chatbot_config.get('lazy_loading', False)
        
        # Initialize with no agent if lazy loading
        if self.lazy_loading:
            self.agent_list = []
            self.current_agent = None
            self.agent_hub = None
        else:
            if isinstance(agent, MultiAgentHub):
                self.agent_list = [agent for agent in agent.nonuser_agents]
                self.agent_hub = agent
            elif isinstance(agent, list):
                self.agent_list = agent
                self.agent_hub = None
            elif agent is not None:
                self.agent_list = [agent]
                self.agent_hub = None
            else:
                self.agent_list = []
                self.agent_hub = None
            
            self.current_agent = self.agent_list[0] if self.agent_list else None

        user_name = chatbot_config.get('user.name', 'user')
        self.user_config = {
            'name': user_name,
            'avatar': chatbot_config.get(
                'user.avatar',
                get_avatar_image(user_name) if 'get_avatar_image' in globals() else None,
            ),
        }

        # Initialize agent config list
        if self.agent_list:
            self.agent_config_list = [{
                'name': agent.name if hasattr(agent, 'name') else 'Assistant',
                'avatar': chatbot_config.get('agent.avatar', None),
                'description': agent.description if hasattr(agent, 'description') else "I'm a helpful assistant.",
            } for agent in self.agent_list]
        else:
            self.agent_config_list = [{
                'name': 'No Model Loaded',
                'avatar': None,
                'description': 'Please select and load a model to begin.',
            }]

        self.input_placeholder = chatbot_config.get('input.placeholder', 'Chat with me~')
        self.prompt_suggestions = chatbot_config.get('prompt.suggestions', [])
        self.verbose = chatbot_config.get('verbose', False)

    def run(self,
            messages: List[Message] = None,
            share: bool = False,
            server_name: str = None,
            server_port: int = None,
            concurrency_limit: int = 10,
            enable_mention: bool = False,
            **kwargs):
        self.run_kwargs = kwargs

        from qwen_agent.gui.gradio import gr, mgr

        customTheme = gr.themes.Default(
            primary_hue=gr.themes.utils.colors.blue,
            radius_size=gr.themes.utils.sizes.radius_none,
        )

        with gr.Blocks(
                css=os.path.join(os.path.dirname(__file__), 'assets/appBot.css') if os.path.exists(os.path.join(os.path.dirname(__file__), 'assets/appBot.css')) else None,
                theme=customTheme,
                title="Qwen2.5 Local Chat"
        ) as demo:
            history = gr.State([])
            model_loaded_state = gr.State(False)

            with gr.Row(elem_classes='container'):
                with gr.Column(scale=4):
                    chatbot = mgr.Chatbot(
                        value=convert_history_to_chatbot(messages=messages) if messages else [],
                        avatar_images=[
                            self.user_config,
                            self.agent_config_list,
                        ],
                        height=600,
                        avatar_image_width=80,
                        flushing=False,
                        show_copy_button=True,
                        latex_delimiters=[{
                            'left': '\\(',
                            'right': '\\)',
                            'display': True
                        }, {
                            'left': '\\[',
                            'right': '\\]',
                            'display': True
                        }]
                    )

                    with gr.Row():
                        input = mgr.MultimodalInput(
                            placeholder=self.input_placeholder, 
                            upload_button_props=dict(
                                file_types=[".pdf", ".docx", ".pptx", ".txt", ".html", ".csv", ".tsv", ".xlsx", ".xls"] + 
                                ["." + ext for ext in common_programming_language_extensions]
                            ),
                            interactive=False  # Disabled until model is loaded
                        )

                with gr.Column(scale=1):
                    # Model configuration section
                    if self.enable_quantization_selector and self.model_manager:
                        gr.Markdown("## 🤖 Model Configuration")
                        
                        quantization_selector = gr.Radio(
                            choices=[
                                ("4-bit + Vector DB (Unlimited docs)", "4bit"),
                                ("8-bit (16k tokens)", "8bit"),
                                ("Full Precision (8k tokens)", "full")
                            ],
                            label="Select Configuration",
                            value="4bit",
                            interactive=True,
                        )
                        
                        load_model_btn = gr.Button(
                            "🚀 Load Model", 
                            variant="primary",
                            size="lg"
                        )
                        
                        model_status = gr.Markdown("""
⚠️ **No Model Loaded**

Please select a configuration and click 'Load Model' to begin.

**Configuration Guide:**
- **4-bit + Vector DB**: Best for long documents (PDFs, books). Uses vector database for unlimited document size.
- **8-bit**: Balanced performance for medium documents.
- **Full Precision**: Best quality for short documents.
                        """)
                        
                        # System resources display
                        gr.Markdown("### 📊 System Resources")
                        resource_display = gr.Markdown(self._get_resource_status())
                        
                        # Manual refresh button (without size parameter for compatibility)
                        refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                        
                    else:
                        quantization_selector = gr.State("4bit")
                        model_status = gr.State("")
                        load_model_btn = gr.State(None)
                        resource_display = gr.State("")
                        refresh_btn = gr.State(None)

                    # Agent info
                    agent_info_block = self._create_agent_info_block()
                    
                    # Clear button
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

                    if self.prompt_suggestions:
                        gr.Examples(
                            label='Example Prompts',
                            examples=self.prompt_suggestions,
                            inputs=[input],
                        )

                # Event handlers
                if self.enable_quantization_selector and self.model_manager:
                    def load_new_model(quantization, _chatbot, _history):
                        """Load model with selected quantization"""
                        try:
                            # Clear chat history
                            _chatbot = []
                            _history.clear()
                            
                            # Update status
                            status = f"⏳ **Loading {quantization.upper()} model...**\n\nThis may take 30-60 seconds depending on your system."
                            yield (
                                status, 
                                _chatbot, 
                                _history, 
                                gr.update(interactive=False),
                                gr.update(interactive=False),
                                False,
                                self._get_resource_status()
                            )
                            
                            # Load the model
                            self.current_agent = self.model_manager.load_model(quantization)
                            
                            if self.current_agent:
                                self.agent_list = [self.current_agent]
                                
                                # Update agent config
                                self.agent_config_list = [{
                                    'name': self.current_agent.name,
                                    'avatar': self.agent_config_list[0].get('avatar'),
                                    'description': self.current_agent.description,
                                }]
                                
                                # Get configuration details
                                limits = self.model_manager.get_current_limits()
                                
                                # Build status message
                                status = f"""✅ **Model Successfully Loaded!**

**Configuration:** {quantization.upper()}
**Mode:** {"Vector Database (Unlimited documents)" if limits.get('use_vector_db') else "Direct Context"}

**Token Limits:**
- Max Context: {limits['max_context_tokens']:,} tokens
- Max Document: {"Unlimited with Vector DB" if limits.get('use_vector_db') else f"{limits['max_ref_token']:,} tokens"}
- Max Response: {limits['max_new_tokens']:,} tokens

{"📚 **Vector Database Active**: Upload any size document! The system will automatically chunk and index it for efficient retrieval." if limits.get('use_vector_db') else ""}

You can now start chatting and uploading documents!"""
                                
                                # Update resource display
                                resource_status = self._get_resource_status()
                                
                                yield (
                                    status, 
                                    _chatbot, 
                                    _history, 
                                    gr.update(interactive=True, value="🔄 Switch Model"),
                                    gr.update(interactive=True, placeholder="Type your message or upload documents..."),
                                    True,
                                    resource_status
                                )
                            else:
                                raise Exception("Failed to load model")
                            
                        except Exception as e:
                            logger.error(f"Error loading model: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            status = f"""❌ **Error Loading Model**

{str(e)}

**Troubleshooting:**
1. Check if you have enough free memory
2. Try closing other applications
3. Try using 4-bit quantization for lower memory usage
4. Ensure CUDA is properly installed (for GPU)"""
                            
                            yield (
                                status, 
                                _chatbot, 
                                _history, 
                                gr.update(interactive=True),
                                gr.update(interactive=False),
                                False,
                                self._get_resource_status()
                            )
                    
                    load_model_btn.click(
                        fn=load_new_model,
                        inputs=[quantization_selector, chatbot, history],
                        outputs=[model_status, chatbot, history, load_model_btn, input, model_loaded_state, resource_display],
                        queue=True,
                    )
                    
                    # Manual refresh button handler
                    def refresh_resources():
                        return self._get_resource_status()
                    
                    refresh_btn.click(
                        fn=refresh_resources,
                        inputs=[],
                        outputs=[resource_display]
                    )

                # Clear chat handler
                def clear_chat(_chatbot, _history):
                    _chatbot = []
                    _history.clear()
                    
                    # Clear vector database if it exists
                    if self.model_manager and hasattr(self.model_manager, 'current_bot'):
                        try:
                            # Clear vector DB collection if using vector search
                            pass  # Vector DB clearing can be added here if needed
                        except:
                            pass
                    
                    return _chatbot, _history
                
                clear_btn.click(
                    fn=clear_chat,
                    inputs=[chatbot, history],
                    outputs=[chatbot, history],
                    queue=False
                )

                # Input handlers with model check
                def check_model_and_add_text(_input, _chatbot, _history, model_loaded):
                    from qwen_agent.gui.gradio import gr
                    
                    # Check if model is loaded
                    if not model_loaded or not self.current_agent:
                        gr.Warning("Please load a model first by clicking 'Load Model'")
                        return gr.update(interactive=True, value=_input), _chatbot, _history, gr.update()
                    
                    # Handle clear command
                    if _input.text == "/clear":
                        _chatbot = []
                        _history.clear()
                        return gr.update(interactive=True, value=""), _chatbot, _history, gr.update()
                    
                    # Normal message processing
                    _history.append({
                        ROLE: USER,
                        CONTENT: [{
                            'text': _input.text
                        }],
                    })

                    if self.user_config.get(NAME):
                        _history[-1][NAME] = self.user_config[NAME]

                    if _input.files:
                        for file in _input.files:
                            if file.mime_type and file.mime_type.startswith('image/'):
                                _history[-1][CONTENT].append({IMAGE: 'file://' + file.path})
                            else:
                                _history[-1][CONTENT].append({FILE: file.path})

                    _chatbot.append([_input, None])

                    return gr.update(interactive=False, value=None), _chatbot, _history, gr.update(interactive=False)

                input_promise = input.submit(
                    fn=check_model_and_add_text,
                    inputs=[input, chatbot, history, model_loaded_state],
                    outputs=[input, chatbot, history, input],
                    queue=False,
                ).then(
                    self.agent_run,
                    [chatbot, history],
                    [chatbot, history],
                ).then(
                    self.flushed, 
                    [model_loaded_state], 
                    [input]
                )

            demo.load(None)

        demo.queue(default_concurrency_limit=concurrency_limit).launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )

    def agent_run(self, _chatbot, _history):
        """Run the agent with proper error handling"""
        if not _history or not self.current_agent:
            yield _chatbot, _history
            return

        if self.verbose:
            logger.info('agent_run input:\n' + pprint.pformat(_history, indent=2))

        num_input_bubbles = len(_chatbot) - 1
        num_output_bubbles = 1
        _chatbot[-1][1] = [None]

        responses = []
        try:
            for responses in self.current_agent.run(_history, **self.run_kwargs):
                if not responses:
                    continue
                if responses[-1][CONTENT] == PENDING_USER_INPUT:
                    logger.info('Interrupted. Waiting for user input!')
                    break

                display_responses = convert_fncall_to_text(responses)
                if not display_responses:
                    continue
                if display_responses[-1][CONTENT] is None:
                    continue

                while len(display_responses) > num_output_bubbles:
                    _chatbot.append([None, [None]])
                    num_output_bubbles += 1

                for i, rsp in enumerate(display_responses):
                    _chatbot[num_input_bubbles + i][1][0] = rsp[CONTENT]

                yield _chatbot, _history

        except Exception as e:
            logger.error(f"Error in agent_run: {e}")
            print_traceback()
            error_msg = f"⚠️ Error: {str(e)}"
            _chatbot[-1][1][0] = error_msg
            yield _chatbot, _history

        if responses:
            for res in responses:
                res['content'] = re.sub(r"\n<summary>input tokens.*</summary>", "", res.get('content', ''))
            _history.extend([res for res in responses if res.get(CONTENT) != PENDING_USER_INPUT])

        yield _chatbot, _history

        if self.verbose:
            logger.info('agent_run response:\n' + pprint.pformat(responses, indent=2))

    def flushed(self, model_loaded):
        from qwen_agent.gui.gradio import gr
        if model_loaded:
            return gr.update(interactive=True)
        else:
            return gr.update(interactive=False)

    def _create_agent_info_block(self):
        from qwen_agent.gui.gradio import gr

        if self.agent_config_list:
            agent_config = self.agent_config_list[0]
        else:
            agent_config = {
                'name': 'No Model Loaded',
                'description': 'Please load a model to begin.',
                'avatar': None
            }

        return gr.HTML(
            format_cover_html(
                bot_name=agent_config['name'],
                bot_description=agent_config['description'],
                bot_avatar=agent_config['avatar'],
            ))
    
    def _get_resource_status(self):
        """Get current system resource status"""
        status_lines = []
        
        # GPU Status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - allocated
            
            gpu_emoji = "🟢" if free > 4 else "🟡" if free > 2 else "🔴"
            status_lines.append(f"{gpu_emoji} **GPU:** {allocated:.1f}/{total:.1f} GB ({free:.1f} GB free)")
        else:
            status_lines.append("🔴 **GPU:** Not available (CPU mode)")
        
        # CPU Memory Status
        memory = psutil.virtual_memory()
        cpu_used = memory.used / 1024**3
        cpu_total = memory.total / 1024**3
        cpu_free = memory.available / 1024**3
        
        cpu_emoji = "🟢" if cpu_free > 8 else "🟡" if cpu_free > 4 else "🔴"
        status_lines.append(f"{cpu_emoji} **RAM:** {cpu_used:.1f}/{cpu_total:.1f} GB ({cpu_free:.1f} GB free)")
        
        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_usage_emoji = "🟢" if cpu_percent < 50 else "🟡" if cpu_percent < 80 else "🔴"
        status_lines.append(f"{cpu_usage_emoji} **CPU:** {cpu_percent:.0f}% usage")
        
        # Model Status
        if self.model_manager and self.model_manager.is_model_loaded():
            status_lines.append(f"✅ **Model:** {self.model_manager.current_quantization or 'Unknown'}")
        else:
            status_lines.append("⚪ **Model:** Not loaded")
        
        return "\n".join(status_lines)