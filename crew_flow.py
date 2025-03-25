from crewai.flow.flow import Flow, listen, start, router
from crewai import Crew, Agent, Task, Process
from crewai.tools import tool
from pydantic import BaseModel
import re
from dotenv import load_dotenv

from src.crew_def.image_generator import image_generator_crew, generate_image

load_dotenv()


# Define structured state for precise control
class UserNotes(BaseModel):
    task_id: str = ""
    is_rendering_related: bool = False
    description: str = ""


@tool
def format_response(task_id: str, is_rendering_related: bool, description: str) -> UserNotes:
    """Format the AI-generated response as a structured Pydantic model."""
    return UserNotes(task_id=task_id, is_rendering_related=is_rendering_related, description=description)


class AdvancedAnalysisFlow(Flow):
    @start()
    def extract_speech_log(self, file_path="UnityUserSpeechLogSample.txt"):
        speech_dict = {}

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                match = re.match(r"\[(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\] User Said: \"(.+)\"", line)
                if match:
                    timestamp, speech = match.groups()
                    speech_dict[timestamp] = speech  # Store in dictionary
        for timestamp, text in speech_dict.items():
            print(f"{timestamp}: {text}")
        return speech_dict

    @listen(extract_speech_log)
    def analyze_with_crew(self, speech_dict):
        # Show crew agency through specialized roles
        secretary = Agent(
            role="Senior Secretary",
            goal="Analyse given information and filter them into image rendering related tasks versus other tasks",
            backstory="You're a meticulous secretary with a keen eye for detail. You're known for your ability to "
                      "summarise notes and filter them into tasks, making it easy for others to "
                      "understand and act on the information you provide.",
            tools=[format_response],
        )

        note_summarising_task = Task(
            description="Generate a response for the given {user_input}, and return it as a structured UserNotes model.",
            expected_output="A JSON response containing a list of the tasks amd their details.",
            agent=secretary
        )

        # Demonstrate crew autonomy
        analysis_crew = Crew(
            agents=[secretary],
            tasks=[note_summarising_task],
            process=Process.sequential,
            verbose=True
        )
        return analysis_crew.kickoff(inputs={"user_input": speech_dict})

    @router(analyze_with_crew)
    def render_images(self):
        import json
        # Show flow control with conditional routing
        output_results = json.loads(self.method_outputs[1].raw)
        if 'tasks' in output_results:
            output_results = output_results['tasks']
        print(output_results)
        for task in output_results:
            if "is_rendering_related" not in task.keys():
                print(f"{task['task_id']}: Not an image rendering task.")
                continue
            if task["is_rendering_related"]:
                print(f"{task['task_id']}: Image rendering task.")
                result = image_generator_crew.kickoff(inputs={"user_input": task["description"]})
                image_prompt = result.raw.split("prompt")[1].split('"')[-2]
                generate_image(image_prompt, image_name=f"{task['task_id']}.png")


flow = AdvancedAnalysisFlow()
flow.kickoff()
