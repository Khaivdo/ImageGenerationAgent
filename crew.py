from crewai import Agent, Crew, Process, Task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from image_generator import generate_image
content = (
    """generate an image with a room that has rainbow color walls""")
string_source = StringKnowledgeSource(content=content)


from pydantic import BaseModel


class ImageGeneration(BaseModel):
    user_input: dict
    prompt: str


@tool
def format_response(user_input: dict, prompt: str) -> ImageGeneration:
    """Format the AI-generated response as a structured Pydantic model."""
    return ImageGeneration(user_input=user_input, prompt=prompt)


@CrewBase
class AgentEnviz():
    """AgentEnviz crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def image_renderer(self) -> Agent:
        return Agent(
            config=self.agents_config['image_renderer'],
            verbose=True, tools=[format_response]
        )

    @task
    def image_rendering_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_rendering_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AgentEnviz crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[string_source],
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
