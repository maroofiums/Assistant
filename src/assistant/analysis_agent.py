from assistant.model import get_model

class AnalysisAgent:

    def __init__(self):
        self.model = get_model()


    def analyze(self, research: str):

        prompt = f"""
        Analyze this research.
        Identify key findings, limitations and conclusions.

        Research:
        {research}
        """ 

        return self.model.invoke(prompt).content