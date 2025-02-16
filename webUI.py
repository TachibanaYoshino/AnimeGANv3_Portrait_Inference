import gradio as gr
import os, cv2, base64, json, requests, argparse
import numpy as np

try:
    from .convert import Conversion, style_list
except:
    from convert import Conversion, style_list

styles_list = style_list
print(styles_list)

def clear_input():
    return None, "USA", 'no', None

def RUN(rgb_img, select_style, use_background_mode):
    if use_background_mode =='yes':
        bg = True
    else:
        bg = False
    _, _, bgr = Conversion(rgb_img[:,:,::-1], select_style, bg)
    return bgr[:,:,::-1]


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""
def main_demo():

    with gr.Blocks(
        title="AnimeGANv3: To produce your own animation.",
        css="""footer {visibility: hidden;} body { background-color: #FFFFFF; margin:10px;}""",
        js=js_func
    ) as demo:

        gr.Markdown("<div align='center'> <h2> AnimeGANv3 webUI </h2> </div> ")
        description = r"""Official online demo for <a href='https://github.com/TachibanaYoshino/AnimeGANv3' target='_blank'><b>AnimeGANv3</b></a>.<br>
        <span style="font-weight: bold;"> The backend algorithm of AnimeGANv3 webUI runs on edge devices. When the number of visits is too large, the computing power of the edge device may not be able to support it.  </span> <br>
        If AnimeGANv3 is helpful, please help to ⭐ the <a href='https://github.com/TachibanaYoshino/AnimeGANv3' target='_blank'>Github Repo</a> and recommend it to your friends. 😊
        """
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem('① Upload'):
                        with gr.Row():
                            input_img = gr.Image(type="numpy", label="Input")

                with gr.Tabs():
                    with gr.TabItem('② Style'):
                        with gr.Row():
                            select_style = gr.Dropdown(choices=[x for x in styles_list], label="Choose the AnimeGANv3 style you want?", value=styles_list[0], )

                with gr.Tabs():
                    with gr.TabItem('③ Include background?'):
                        with gr.Row():
                            use_background_mode = gr.Radio(["yes", "no"], label="Convert background?", value="yes", type="value")


            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem('④ Result'):
                        with gr.Row():
                            ouput_result = gr.Image(type="numpy", label="Output image")
                        with gr.Row():
                            Synthesize_btn = gr.Button(value="Synthesize", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Group():
            with gr.Row():
                gr.Examples(
                      label="Input Examples",
                      examples=[['data/Kobe.png', 'Pixar', "yes"],
                                ['data/lqd.jpg', 'Arcane', "yes"],
                                ['data/segA00001.jpg',   'Trump2.0', "yes"],
                                ['data/segA00003.jpg', 'Sketch-0', "yes"],
                                ['data/segA00004.jpg', 'Disney2.0', "no"],
                                ['data/segA00007.jpg', 'Kpop', "no"],
                                ['data/segA00006.jpg', 'Comic', "yes"]],
                      inputs=[input_img, select_style, use_background_mode],
                      outputs=[ouput_result],
                      fn=RUN,
                      run_on_click=False
                  )

        article = r"""
        ----------
        [![Github](https://img.shields.io/github/stars/TachibanaYoshino/AnimeGANv3?logo=githubsponsors&logoColor=#EA4AAA)](https://github.com/TachibanaYoshino/AnimeGANv3) <br>
        ## License  
        This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications. Permission is granted to use the AnimeGANv3 given that you agree to my license terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.
        """
        gr.Markdown(article)

        Synthesize_btn.click(fn=RUN, inputs=[input_img, select_style, use_background_mode], outputs=[ouput_result])
        clear_btn.click(fn=clear_input, inputs=[], outputs=[input_img, select_style, use_background_mode, ouput_result])
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description='start service')
    parser.add_argument('--port', '-p', default='9999', help='port')
    parser.add_argument('--ip', '-i', default='0.0.0.0', help='port')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    port = int(args.port)
    ip = args.ip
    print(ip, port)

    demo = main_demo()
    demo.launch( server_name=ip, server_port=port, share=False)