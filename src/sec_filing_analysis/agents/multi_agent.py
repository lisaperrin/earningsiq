from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from ..rag.query_engine import RAGQueryEngine
from ..data.price_data import PriceDataFetcher
from ..utils.logger import log


class AgentRole(Enum):
    QUERY_DECOMPOSER = "query_decomposer"
    SECTION_SPECIALIST = "section_specialist"
    FINANCIAL_ANALYST = "financial_analyst"
    RISK_ANALYST = "risk_analyst"
    PRICE_ANALYST = "price_analyst"
    SYNTHESIS = "synthesis"


@dataclass
class AgentTask:
    task_id: str
    description: str
    agent_role: AgentRole
    parameters: Dict
    priority: int = 1


@dataclass
class AgentResponse:
    task_id: str
    agent_role: AgentRole
    result: any
    metadata: Dict
    success: bool = True


class BaseAgent(ABC):
    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResponse:
        pass


class QueryDecomposerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Query Decomposer", AgentRole.QUERY_DECOMPOSER)

    def execute(self, task: AgentTask) -> AgentResponse:
        query = task.parameters.get("query", "")
        ticker = task.parameters.get("ticker")
        year = task.parameters.get("year")

        sub_tasks = self._decompose_query(query, ticker, year)

        return AgentResponse(
            task_id=task.task_id,
            agent_role=self.role,
            result=sub_tasks,
            metadata={"original_query": query}
        )

    def _decompose_query(self, query: str, ticker: Optional[str], year: Optional[int] = None) -> List[AgentTask]:
        query_lower = query.lower()
        sub_tasks = []

        if "price" in query_lower or "stock" in query_lower or "reaction" in query_lower:
            sub_tasks.append(AgentTask(
                task_id="price_analysis",
                description="Analyze price reaction",
                agent_role=AgentRole.PRICE_ANALYST,
                parameters={"ticker": ticker, "query": query, "year": year}
            ))

        if "risk" in query_lower or "concern" in query_lower or "challenge" in query_lower:
            sub_tasks.append(AgentTask(
                task_id="risk_analysis",
                description="Analyze risk factors",
                agent_role=AgentRole.RISK_ANALYST,
                parameters={"ticker": ticker, "query": query, "section": "RISK_FACTORS", "year": year}
            ))

        if "revenue" in query_lower or "earnings" in query_lower or "guidance" in query_lower:
            sub_tasks.append(AgentTask(
                task_id="financial_analysis",
                description="Analyze financial metrics",
                agent_role=AgentRole.FINANCIAL_ANALYST,
                parameters={"ticker": ticker, "query": query, "section": "MD&A", "year": year}
            ))

        if not sub_tasks:
            sub_tasks.append(AgentTask(
                task_id="general_analysis",
                description="General financial analysis",
                agent_role=AgentRole.FINANCIAL_ANALYST,
                parameters={"ticker": ticker, "query": query, "year": year}
            ))

        return sub_tasks


class SectionSpecialistAgent(BaseAgent):
    def __init__(self, rag_engine: RAGQueryEngine, section: str):
        super().__init__(f"{section} Specialist", AgentRole.SECTION_SPECIALIST)
        self.rag_engine = rag_engine
        self.section = section

    def execute(self, task: AgentTask) -> AgentResponse:
        query = task.parameters.get("query")
        ticker = task.parameters.get("ticker")

        try:
            result = self.rag_engine.query(
                query,
                ticker=ticker,
                section=self.section,
                top_k=5
            )

            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=result,
                metadata={"section": self.section, "ticker": ticker}
            )

        except Exception as e:
            log.error(f"Section specialist agent failed: {e}")
            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=None,
                metadata={"error": str(e)},
                success=False
            )


class FinancialAnalystAgent(BaseAgent):
    def __init__(self, rag_engine: RAGQueryEngine):
        super().__init__("Financial Analyst", AgentRole.FINANCIAL_ANALYST)
        self.rag_engine = rag_engine

    def execute(self, task: AgentTask) -> AgentResponse:
        query = task.parameters.get("query")
        ticker = task.parameters.get("ticker")
        section = task.parameters.get("section", "MD&A")
        year = task.parameters.get("year")

        try:
            result = self.rag_engine.query(
                query,
                ticker=ticker,
                section=section,
                year=year,
                top_k=5
            )

            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=result,
                metadata={"ticker": ticker, "section": section}
            )

        except Exception as e:
            log.error(f"Financial analyst agent failed: {e}")
            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=None,
                metadata={"error": str(e)},
                success=False
            )


class RiskAnalystAgent(BaseAgent):
    def __init__(self, rag_engine: RAGQueryEngine):
        super().__init__("Risk Analyst", AgentRole.RISK_ANALYST)
        self.rag_engine = rag_engine

    def execute(self, task: AgentTask) -> AgentResponse:
        query = task.parameters.get("query")
        ticker = task.parameters.get("ticker")
        year = task.parameters.get("year")

        try:
            result = self.rag_engine.query(
                query,
                ticker=ticker,
                section="RISK_FACTORS",
                year=year,
                top_k=5
            )

            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=result,
                metadata={"ticker": ticker}
            )

        except Exception as e:
            log.error(f"Risk analyst agent failed: {e}")
            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=None,
                metadata={"error": str(e)},
                success=False
            )


class PriceAnalystAgent(BaseAgent):
    def __init__(self, price_fetcher: PriceDataFetcher):
        super().__init__("Price Analyst", AgentRole.PRICE_ANALYST)
        self.price_fetcher = price_fetcher

    def execute(self, task: AgentTask) -> AgentResponse:
        ticker = task.parameters.get("ticker")

        try:
            price_data = self.price_fetcher.get_stock_data(ticker, period="1y")
            fundamentals = self.price_fetcher.get_fundamentals(ticker)

            result = {
                "price_data": price_data,
                "fundamentals": fundamentals,
                "ticker": ticker
            }

            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=result,
                metadata={"ticker": ticker}
            )

        except Exception as e:
            log.error(f"Price analyst agent failed: {e}")
            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=None,
                metadata={"error": str(e)},
                success=False
            )


class SynthesisAgent(BaseAgent):
    def __init__(self, rag_engine: RAGQueryEngine):
        super().__init__("Synthesis Agent", AgentRole.SYNTHESIS)
        self.rag_engine = rag_engine

    def execute(self, task: AgentTask) -> AgentResponse:
        agent_responses = task.parameters.get("agent_responses", [])
        original_query = task.parameters.get("original_query", "")

        try:
            synthesis = self._synthesize_responses(agent_responses, original_query)

            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=synthesis,
                metadata={"num_responses": len(agent_responses)}
            )

        except Exception as e:
            log.error(f"Synthesis agent failed: {e}")
            return AgentResponse(
                task_id=task.task_id,
                agent_role=self.role,
                result=None,
                metadata={"error": str(e)},
                success=False
            )

    def _synthesize_responses(self, responses: List[AgentResponse], query: str) -> Dict:
        synthesis_parts = []

        for response in responses:
            if response.success and response.result:
                if response.agent_role == AgentRole.FINANCIAL_ANALYST:
                    synthesis_parts.append(f"Financial Analysis:\n{response.result.get('answer', '')}")
                elif response.agent_role == AgentRole.RISK_ANALYST:
                    synthesis_parts.append(f"Risk Analysis:\n{response.result.get('answer', '')}")
                elif response.agent_role == AgentRole.PRICE_ANALYST:
                    fundamentals = response.result.get('fundamentals', {})
                    synthesis_parts.append(f"Price & Fundamentals:\nP/E: {fundamentals.get('pe_ratio')}, Sector: {fundamentals.get('sector')}")

        final_synthesis = "\n\n".join(synthesis_parts)

        return {
            "query": query,
            "synthesis": final_synthesis,
            "component_responses": responses
        }


class MultiAgentOrchestrator:
    def __init__(self, rag_engine: RAGQueryEngine, price_fetcher: PriceDataFetcher):
        self.rag_engine = rag_engine
        self.price_fetcher = price_fetcher

        self.agents = {
            AgentRole.QUERY_DECOMPOSER: QueryDecomposerAgent(),
            AgentRole.FINANCIAL_ANALYST: FinancialAnalystAgent(rag_engine),
            AgentRole.RISK_ANALYST: RiskAnalystAgent(rag_engine),
            AgentRole.PRICE_ANALYST: PriceAnalystAgent(price_fetcher),
            AgentRole.SYNTHESIS: SynthesisAgent(rag_engine),
        }

    def execute_query(self, query: str, ticker: Optional[str] = None, year: Optional[int] = None) -> Dict:
        log.info(f"Executing multi-agent query: {query}")

        decomposer_task = AgentTask(
            task_id="decompose",
            description="Decompose query",
            agent_role=AgentRole.QUERY_DECOMPOSER,
            parameters={"query": query, "ticker": ticker, "year": year}
        )

        decomposer_response = self.agents[AgentRole.QUERY_DECOMPOSER].execute(decomposer_task)
        sub_tasks = decomposer_response.result

        agent_responses = []
        for sub_task in sub_tasks:
            agent = self.agents.get(sub_task.agent_role)
            if agent:
                response = agent.execute(sub_task)
                agent_responses.append(response)
                log.info(f"Completed task: {sub_task.task_id} by {sub_task.agent_role.value}")

        synthesis_task = AgentTask(
            task_id="synthesize",
            description="Synthesize results",
            agent_role=AgentRole.SYNTHESIS,
            parameters={
                "agent_responses": agent_responses,
                "original_query": query
            }
        )

        synthesis_response = self.agents[AgentRole.SYNTHESIS].execute(synthesis_task)

        return {
            "query": query,
            "ticker": ticker,
            "sub_tasks": [task.description for task in sub_tasks],
            "agent_responses": agent_responses,
            "final_synthesis": synthesis_response.result
        }
