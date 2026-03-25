"""Tests for code citation schema validation."""
from reasoners.navigator import CodeCitation, AnswerWithDrilldown


class TestCodeCitation:
    def test_basic_citation(self):
        c = CodeCitation(file="main.py", start_line=10, snippet="entry point")
        assert c.file == "main.py"
        assert c.start_line == 10
        assert c.end_line is None
        assert c.symbol == ""

    def test_full_citation(self):
        c = CodeCitation(
            file="src/auth.py",
            start_line=25,
            end_line=40,
            symbol="verify_token",
            snippet="JWT verification logic",
        )
        assert c.end_line == 40
        assert c.symbol == "verify_token"

    def test_citation_in_answer(self):
        answer = AnswerWithDrilldown(
            answer="The auth uses JWT [1]",
            citations=[
                CodeCitation(file="auth.py", start_line=10, symbol="verify"),
            ],
            relevant_files=["auth.py"],
            confidence="high",
            follow_up=["How is the token refreshed?"],
        )
        assert len(answer.citations) == 1
        assert answer.citations[0].file == "auth.py"

    def test_empty_citations(self):
        answer = AnswerWithDrilldown(
            answer="No specific code references needed",
            citations=[],
            relevant_files=[],
            confidence="medium",
            follow_up=[],
        )
        assert len(answer.citations) == 0

    def test_multiple_citations(self):
        answer = AnswerWithDrilldown(
            answer="Routes [1] call services [2] which hit the DB [3]",
            citations=[
                CodeCitation(file="routes.py", start_line=5, symbol="get_users"),
                CodeCitation(file="services/user.py", start_line=20, symbol="UserService"),
                CodeCitation(file="db/models.py", start_line=1, end_line=30, symbol="User"),
            ],
            relevant_files=["routes.py", "services/user.py", "db/models.py"],
            confidence="high",
            follow_up=[],
        )
        assert len(answer.citations) == 3
        assert answer.citations[2].end_line == 30
