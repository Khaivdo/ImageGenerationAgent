from crewai.tools import tool

from crewai import Agent, Task

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()


def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())


def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  # Execution is done
        else:
            # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
            # bytesIO = BytesIO(out[8:])
            # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
            continue  # previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images


prompt_text = """
{
  "3": {
    "inputs": {
      "seed": 1037527948124596,
      "steps": 20,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "31",
        0
      ],
      "positive": [
        "35",
        0
      ],
      "negative": [
        "35",
        1
      ],
      "latent_image": [
        "35",
        2
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "7": {
    "inputs": {
      "text": "",
      "clip": [
        "34",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Negative Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "32",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "17": {
    "inputs": {
      "image": "2025-02-15_10-09-584.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "18": {
    "inputs": {
      "low_threshold": 0.25,
      "high_threshold": 0.3,
      "image": [
        "17",
        0
      ]
    },
    "class_type": "Canny",
    "_meta": {
      "title": "Canny"
    }
  },
  "19": {
    "inputs": {
      "images": [
        "18",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "23": {
    "inputs": {
      "text": "A modern kitchen with rainy usonian architecture, cozy bright warm lighting.",
      "clip": [
        "34",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Positive Prompt)"
    }
  },
  "26": {
    "inputs": {
      "guidance": 30,
      "conditioning": [
        "23",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "31": {
    "inputs": {
      "unet_name": "flux1CannyDevFp8_v10.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "32": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "34": {
    "inputs": {
      "clip_name1": "clip_l.safetensors",
      "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
      "type": "flux",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "35": {
    "inputs": {
      "positive": [
        "26",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "vae": [
        "32",
        0
      ],
      "pixels": [
        "18",
        0
      ]
    },
    "class_type": "InstructPixToPixConditioning",
    "_meta": {
      "title": "InstructPixToPixConditioning"
    }
  }
}
"""


def generate_image(image_prompt: str, image_name: str = None) -> str:
    print(f"INFO:: {image_name}: '{image_prompt}'")
    """Generate image given a prompt."""
    prompt = json.loads(prompt_text)
    # # set the text prompt for our positive CLIPTextEncode
    prompt["23"]["inputs"]["text"] = image_prompt
    if image_name:
        prompt["17"]["inputs"]["image"] = image_name


    # set the seed for our KSampler node
    # prompt["3"]["inputs"]["seed"] = 5

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt)
    ws.close()
    return "Image generated."


class ImageGeneration(BaseModel):
    user_input: dict
    prompt: str


@tool
def format_response(user_input: dict, prompt: str) -> ImageGeneration:
    """Format the AI-generated response as a structured Pydantic model."""
    return ImageGeneration(user_input=user_input, prompt=prompt)


agent = Agent(
    role="Image Renderer",
    goal="Generate Architecture and Construction related prompts and return structured output",
    backstory="You are an expert in converting information into prompts to generate image with a given input image.You're known for your ability to turn vague information into clear and concise prompts, making it easy for other image generation models, especially Flux based models to understand and act on the information you provide.",
    tools=[format_response],  # Attach the tool to the agent
    verbose=True
)


task = Task(
    description="Generate a response for the given {user_input} and return it as a structured ImageGeneration model.",
    expected_output="A JSON response containing the original input and the AI's generated prompt.",
    agent=agent
)

from crewai import Crew, Process


image_generator_crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential
)


# # Provide an input prompt and get structured output
# result = crew.kickoff(inputs={"user_input": "Give some renders of the kitchen area, preferably master chef kitchen style, in rainbow color."})
#
# # print(result)
# image_prompt = result.raw.split("prompt")[1].split('"')[-2]
# generate_image(image_prompt)
