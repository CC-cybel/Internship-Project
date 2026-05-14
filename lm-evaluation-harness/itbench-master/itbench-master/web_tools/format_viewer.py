import gradio as gr
import re

def format_text(input_text):
    if not input_text:
        return ""
    
    # Replace literal \n with actual newlines to ensure proper formatting
    text = input_text.replace('\\n', '\n')
    
    # Use regex to add a newline before every [第X轮] tag to make sure they start on a new line
    # We use a positive lookahead to find the tag without consuming it
    # We only add a newline if it's not already preceded by one
    formatted_text = re.sub(r'(?<!\n)(\[第\d+轮\])', r'\n\1', text)
    
    return formatted_text

# Create the Gradio interface
with gr.Blocks(title="对话历史格式化查看器") as app:
    gr.Markdown("# 对话历史化查看器")
    gr.Markdown("将一长串包含 `\\n` 和 `[第X轮]` 的纯文本粘贴到左侧，右侧会自动帮你排版、换行，方便阅读。")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="输入原始文本 (Raw Text)",
                lines=20,
                placeholder="粘贴你的字符串在这里...\n例如: system: 角色设定...\\n\\n[第1轮]user: ...",
                max_lines=30
            )
            format_btn = gr.Button("格式化 (Format)", variant="primary")
            
        with gr.Column():
            # Use Textbox instead of Markdown to preserve exact spacing and newlines easily
            output_box = gr.Textbox(
                label="格式化结果 (Formatted Result)",
                lines=20,
                max_lines=30,
                interactive=False
            )
            
    # Trigger formatting when button is clicked or text changes
    format_btn.click(fn=format_text, inputs=input_box, outputs=output_box)
    input_box.change(fn=format_text, inputs=input_box, outputs=output_box)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7861, share=False)
