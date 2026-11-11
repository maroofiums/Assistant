from assistant.analysis_agent import AnalysisAgent
from assistant.research_agent import ResearchAgent

class MultiAgentSystem:

    def __init__(self):
        self.researcher = ResearchAgent()
        self.analyst = AnalysisAgent()

    def run(self, question: str):


        research = self.researcher.research(question)

        analysis = self.analyst.analyze(
            str(research)
        )

        return {
            "research": research,
            "analysis": analysis
        }

