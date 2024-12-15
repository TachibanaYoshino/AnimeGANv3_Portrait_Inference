import gradio as gr
import os, cv2, base64, json, requests, argparse
import numpy as np

try:
    from .convert import Conversion
except:
    from convert import Conversion


# def Conversion(img, style, background):
#     def base642mat(imgBase64):
#         img_data = base64.b64decode(imgBase64)
#         bs = np.asarray(bytearray(img_data), dtype='uint8')
#         mat = cv2.imdecode(bs, cv2.IMREAD_COLOR)  # RGB2BGR
#         mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)  # BGR2RGB
#         return mat
#
#     def cv2_to_b64(img):
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB2BGR
#         image = cv2.imencode('.jpg', img, )[1]  # BGR2RGB
#         base64_data = str(base64.b64encode(image))[2:-1]
#         return base64_data
#
#     headers = {'content-type': "application/json", }
#     port = '8080'
#     ip = '127.0.0.1'
#     # ip = '192.168.1.12'
#     url = f'http://{ip}:{port}/AnimeGANv3/face_style'  #  url
#     base64Data = cv2_to_b64(img) #RGB
#     data = {'RequestID': 'a' * 10, "Inputs": {'b64data': base64Data, 'style': style, 'if_bg': background}}
#     response = requests.post(url, json=json.loads(json.dumps(data)), headers=headers)
#     json_result = response.json()
#     if json_result['Status'] == 0:
#         img= base642mat(json_result["Outputs"]["data"])
#         return img

styles_list =[
    "USA", "USA2", "Comic",  "Cute",  "8bit",  "Arcane", "Pixar",  "Kpop", "Sketch-0", "Nordic_myth1", "Nordic_myth2", "Trump2.0", "Disney2.0"
]

def clear_input():
    return None, None, 'no', None


def main_demo():

    with gr.Blocks(
        title="AnimeGANv3: To produce your own animation.",
        css="""footer {visibility: hidden;} body { background-color: #FFFFFF; margin:10px;}""",

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
                      examples=[['data/1_out.jpg', 'Pixar', "yes"],
                                ['data/7_out.jpg', 'Arcane', "yes"],
                                ['data/120.jpg',   'Trump2.0', "yes"],
                                ['data/15566.jpg', 'Sketch-0', "yes"],
                                ['data/23034.jpg', 'Disney2.0', "no"],
                                ['data/52014.jpg', 'Kpop', "no"],
                                ['data/Hamabe.jpg', 'Comic', "yes"]],
                      inputs=[input_img, select_style, use_background_mode],
                      outputs=[ouput_result],
                      fn=Conversion,
                      run_on_click=False
                  )

        article = r"""
        ----------
        [![Github](https://img.shields.io/github/stars/TachibanaYoshino/AnimeGANv3?logo=githubsponsors&logoColor=#EA4AAA)](https://github.com/TachibanaYoshino/AnimeGANv3) <br>
        ## License  
        This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications. Permission is granted to use the AnimeGANv3 given that you agree to my license terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.
        """
        gr.Markdown(article)

        Synthesize_btn.click(fn=Conversion, inputs=[input_img, select_style, use_background_mode], outputs=[ouput_result])
        clear_btn.click(fn=clear_input, inputs=[], outputs=[input_img, select_style, use_background_mode, ouput_result])
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description='start service')
    parser.add_argument('--port', '-p', default='9999', help='port')
    parser.add_argument('--IP', '-i', default='0.0.0.0', help='port')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    port = int(args.port)
    ip = args.IP
    print(ip, port)

    demo = main_demo()
    demo.launch(max_threads=2, server_name=ip, server_port=port, share=True)