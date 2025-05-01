"""
Brain Systems Module

This module provides implementation of high-level brain systems that implement
biological functions like thalamic gating, basal ganglia action selection,
cerebellar motor correction, and autonomic regulation.
"""

from brain.systems.thalamic_gating import ThalamicGating
from brain.systems.basal_ganglia import BasalGangliaLoop
from brain.systems.cerebellum import CerebellarCorrection
from brain.systems.autonomic import AutonomicSystem

__all__ = [
    'ThalamicGating',
    'BasalGangliaLoop',
    'CerebellarCorrection',
    'AutonomicSystem',
] 