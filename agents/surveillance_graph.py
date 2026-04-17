from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END

from agents.abnormal_activity.abnormal_activity_agent import AbnormalActivityAgent
from agents.crowd_density_agent import CrowdDensityAgent
from agents.panic_detection_agent import PanicDetectionAgent
from agents.suspicious_object_agent import SuspiciousObjectAgent
from agents.risk_evaluation_agent import RiskEvaluationAgent
from agents.decision_alert_agent import DecisionAlertAgent
from shared_processing.detector import ObjectDetector
from shared_processing.tracker import ObjectTracker
from shared_processing.optical_flow import OpticalFlowEstimator
from shared_processing.scene_state import SceneState
from shared_processing.trajectory_manager import TrajectoryManager


class SurveillanceState(TypedDict, total=False):
    frame: Any
    frame_id: int
    timestamp: float
    scene_state: SceneState
    abnormal_result: Dict[str, Any] | None
    crowd_result: Dict[str, Any]
    panic_result: Dict[str, Any]
    suspicious_result: Dict[str, Any]
    risk_result: Dict[str, Any]
    decision_result: Dict[str, Any]


class SurveillanceGraphRunner:
    def __init__(self, frame_width, frame_height, model_path):
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.trajectory_manager = TrajectoryManager()
        self.flow_estimator = OpticalFlowEstimator()

        self.abnormal_agent = AbnormalActivityAgent(
            frame_width=frame_width,
            frame_height=frame_height,
            model_path=model_path,
        )
        self.crowd_agent = CrowdDensityAgent()
        self.panic_agent = PanicDetectionAgent()
        self.suspicious_agent = SuspiciousObjectAgent()
        self.risk_agent = RiskEvaluationAgent()
        self.decision_agent = DecisionAlertAgent()

        # LangChain runnable wrappers for agent execution.
        self.abnormal_chain = RunnableLambda(self.abnormal_agent.process)
        self.crowd_chain = RunnableLambda(self.crowd_agent.process)
        self.panic_chain = RunnableLambda(self.panic_agent.process)
        self.suspicious_chain = RunnableLambda(self.suspicious_agent.process)
        self.risk_chain = RunnableLambda(lambda x: self.risk_agent.process(**x))
        self.decision_chain = RunnableLambda(lambda x: self.decision_agent.process(**x))

        self.graph = self._build_graph()

    def _shared_processing_node(self, state: SurveillanceState):
        frame = state["frame"]
        frame_id = state["frame_id"]
        timestamp = state["timestamp"]

        detections = self.detector.detect(frame)
        tracked = self.tracker.update(detections, frame)
        tracked = self.trajectory_manager.update(tracked)
        flow = self.flow_estimator.compute(frame)

        scene_state = SceneState(
            frame_id=frame_id,
            timestamp=timestamp,
            objects=tracked,
            optical_flow=flow,
            detections=detections,
        )
        return {"scene_state": scene_state}

    def _abnormal_node(self, state: SurveillanceState):
        result = self.abnormal_chain.invoke(state["scene_state"])
        return {"abnormal_result": result}

    def _crowd_node(self, state: SurveillanceState):
        result = self.crowd_chain.invoke(state["scene_state"])
        return {"crowd_result": result}

    def _panic_node(self, state: SurveillanceState):
        result = self.panic_chain.invoke(state["scene_state"])
        return {"panic_result": result}

    def _suspicious_node(self, state: SurveillanceState):
        result = self.suspicious_chain.invoke(state["scene_state"])
        return {"suspicious_result": result}

    def _risk_node(self, state: SurveillanceState):
        abnormal = state.get("abnormal_result")
        if abnormal is None:
            abnormal = {"abnormal": False, "score": 0.0}

        result = self.risk_chain.invoke(
            {
                "abnormal_result": abnormal,
                "crowd_result": state["crowd_result"],
                "panic_result": state["panic_result"],
                "suspicious_result": state["suspicious_result"],
            }
        )
        return {"risk_result": result}

    def _decision_node(self, state: SurveillanceState):
        abnormal = state.get("abnormal_result")
        if abnormal is None:
            abnormal = {"abnormal": False, "score": 0.0}

        result = self.decision_chain.invoke(
            {
                "risk_result": state["risk_result"],
                "abnormal_result": abnormal,
                "panic_result": state["panic_result"],
                "suspicious_result": state["suspicious_result"],
            }
        )
        return {"decision_result": result}

    def _build_graph(self):
        workflow = StateGraph(SurveillanceState)

        workflow.add_node("shared_processing", self._shared_processing_node)
        workflow.add_node("abnormal_activity", self._abnormal_node)
        workflow.add_node("crowd_density", self._crowd_node)
        workflow.add_node("panic_detection", self._panic_node)
        workflow.add_node("suspicious_object", self._suspicious_node)
        workflow.add_node("risk_evaluation", self._risk_node)
        workflow.add_node("decision_alert", self._decision_node)

        workflow.add_edge(START, "shared_processing")

        workflow.add_edge("shared_processing", "abnormal_activity")
        workflow.add_edge("shared_processing", "crowd_density")
        workflow.add_edge("shared_processing", "panic_detection")
        workflow.add_edge("shared_processing", "suspicious_object")

        workflow.add_edge("abnormal_activity", "risk_evaluation")
        workflow.add_edge("crowd_density", "risk_evaluation")
        workflow.add_edge("panic_detection", "risk_evaluation")
        workflow.add_edge("suspicious_object", "risk_evaluation")

        workflow.add_edge("risk_evaluation", "decision_alert")
        workflow.add_edge("decision_alert", END)

        return workflow.compile()

    def run_frame(self, frame, frame_id, timestamp):
        input_state: SurveillanceState = {
            "frame": frame,
            "frame_id": frame_id,
            "timestamp": timestamp,
        }
        return self.graph.invoke(input_state)
