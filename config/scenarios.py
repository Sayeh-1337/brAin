"""
Scenario configurations for VizDoom environments
"""

# Define scenario configurations
SCENARIO_CONFIGS = {
    "basic": {
        "description": "Basic navigation and shooting",
        "skills": ["target shooting", "aiming"],
        "difficulty": "beginner",
        "wad_file": "basic.wad",
        "optimal_params": {
            "hd_dim": 500,
            "learning_rate": 0.01,
            "ca_width": 20,
            "ca_height": 15
        }
    },
    "defend_center": {
        "description": "Defend against enemies coming from all sides",
        "skills": ["precise shooting", "threat awareness", "360° awareness"],
        "difficulty": "intermediate",
        "wad_file": "defend_the_center.wad",
        "optimal_params": {
            "hd_dim": 600,
            "learning_rate": 0.01,
            "ca_width": 20,
            "ca_height": 15
        }
    },
    "deadly_corridor": {
        "description": "Navigate corridor while eliminating enemies",
        "skills": ["path finding", "enemy elimination", "resource management"],
        "difficulty": "advanced",
        "wad_file": "deadly_corridor.wad",
        "optimal_params": {
            "hd_dim": 800,
            "learning_rate": 0.005,
            "ca_width": 25,
            "ca_height": 20
        }
    },
    "health_gathering": {
        "description": "Find and collect health packs to survive",
        "skills": ["navigation", "resource finding", "prioritization"],
        "difficulty": "intermediate",
        "wad_file": "health_gathering.wad",
        "optimal_params": {
            "hd_dim": 600,
            "learning_rate": 0.01,
            "ca_width": 20,
            "ca_height": 15
        }
    },
    "defend_line": {
        "description": "Defend a line against approaching enemies",
        "skills": ["target prioritization", "threat assessment", "shooting accuracy"],
        "difficulty": "intermediate",
        "wad_file": "defend_the_line.wad",  
        "optimal_params": {
            "hd_dim": 600,
            "learning_rate": 0.01,
            "ca_width": 20,
            "ca_height": 15
        }
    }
} 