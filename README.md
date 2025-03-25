## AI agent workflow ## 

### Target ### 
Summarize, filter tasks and carry out image rendering.

### Implementation ###
- Read user notes (e.g. a .txt file with all users notes in this case)
- Extract and filtering tasks based on user notes.

### Pipeline design ###
![AI_agent_flow](https://github.com/user-attachments/assets/7f069b23-cf98-4a2f-9cd3-3f0725c7ce86)


Tasks are then broken down into:
- Image rendering related tasks
- Other tasks

### Image generation ###  
For tasks that require image rendering, AI agent will produce enhanced prompts based on given request, then pass both prompt and sample image to a ComfyUI workflow API to generate image.

### Result ### 
[![Watch the video](https://github.com/user-attachments/assets/ce2351a0-0c08-477c-92a6-a3449ad34d45)](https://www.canva.com/design/DAGiu2-HZzY/3ORq-Yog5FzlnzGbrMPDiQ/watch?utm_content=DAGiu2-HZzY&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h2d691d8b4c)
