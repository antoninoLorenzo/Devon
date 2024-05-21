import unittest

from devon_agent.session import SessionArguments, Session
from devon_agent.agents.default.agent import TaskAgent, PlanningAgent
from devon_agent.agents.default.ollama_prompts import parse_response


class PlanningAgentTest(unittest.TestCase):
    def setUp(self):
        mock_agent = TaskAgent(model='gpt4-o', name='test_agent')

        se_args = SessionArguments(
            name='test_session',
            path='../../../../',
            user_input='Hello World'
        )
        self.session = Session(se_args, mock_agent)
        self.test_queries = [
            {'role': 'user', 'content': 'search the file test.txt'},
            {'role': 'user', 'content': 'Create a Python script called hello.py with content print("Hello, World!")'},
            {'role': 'user', 'content': 'Could you please open the file existing_file.txt?'}
        ]

    def test_forward_phi3(self):
        agent = PlanningAgent('phi3')
        for q in self.test_queries:
            thought, action, output = agent.forward(q['content'], self.session)
            err_msg = f"OllamaModel Test: wrong response format"

            self.assertIsNotNone(thought, err_msg)
            self.assertIsNotNone(action, err_msg)

    def test_run_phi3(self):
        pass


if __name__ == "__main__":
    unittest.main()
