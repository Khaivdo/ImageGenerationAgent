## AI agent workflow ## 

### Target ### 
Summarize, filter tasks and carry out image rendering.

### Implementation ###
- A function to preprocess texts from user notes (e.g. a .txt file with all users notes in this case).
- AI crew:
  - Agent 1: Secretary
    - Extract and filtering tasks based on user notes. Tasks are defined as "Image Rendering" vs "Other tasks".
  - Agent 2: Prompt Enhancer
    - Inherit tasks breakdown from Agent 1.
    - Optimise prompts for tasks related to image rendering, return a structured outputs with tasks ID and task prompt (for image rendering tasks).
- Image Rendering: A ComfyUI API to take input image based on task ID and generate a new image using Flux model + optimised prompt, then save output image to local dir.

### Pipeline design ###
![AI_agent_flow](https://github.com/user-attachments/assets/7f069b23-cf98-4a2f-9cd3-3f0725c7ce86)




### Image generation ###  
For tasks that require image rendering, AI agent will produce enhanced prompts based on given request, then pass both prompt and sample image to a ComfyUI workflow API to generate image.

### Result ### 
[![Watch the video](https://github.com/user-attachments/assets/ce2351a0-0c08-477c-92a6-a3449ad34d45)](https://www.canva.com/design/DAGiu2-HZzY/3ORq-Yog5FzlnzGbrMPDiQ/watch?utm_content=DAGiu2-HZzY&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h2d691d8b4c)
