# 🧠 Curriculum Learning in POMDPs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/DL-PyTorch-orange)
![Reinforcement Learning](https://img.shields.io/badge/Method-PPO%20%2B%20LSTM-purple)

> **Short Summary:** This project implements a Reinforcement Learning agent capable of solving complex, partially observable environments (MiniGrid) by using **Curriculum Learning** and **Memory (LSTM)**.

## 🎥 The Agent in Action

![Demo des Agenten](assets/Big-multiroom.gif)

## 🧐 The Challenge

The agent has to find the green goal square in maze-like environments:
1.  **Partial Observability (POMDP):** The agent only sees a tiny cone in front of it (7x7 pixels). It has to "remember" the map.
2.  **Sparse Rewards:** Finding the goal by accident is nearly impossible in hard levels (Big Multiroom).
3.  **Memory Requirement:** Without looking back, the agent gets lost.

## 💡 Conquering Problems

### 1. Curriculum Learning ( The "School" Method)
Instead of forcing the agent to solve the hardest maze immediately, we can use Curriculum Learning to split the task into simpler actions.
This yields many benefits among which are:
* **Sparse Reward:** In most big mazes the action sequence of the agent has to be extremely specific in order to see any reward at all.
In an environment Like Big-multiroom the right chain of actions without reward practically never happens, the agent will not ever learn
a strategy. This is especially true if you consider random seeds in the initialization of the environment, because the observations
are always something unexpected, something new. Fixed seeds on the other hand, lead to poor generalization over multiple environments 
of the same kind. By splitting the environment into its subtasks (walking through doors to get to reward) helps assigning reward to actions,
which might be essential for success, but at least helps boost convergence speed.

* **Partial Observability:** In Partially observable Markov Decission Processes (POMDP's) the crucial assumption of being able to always
select the perfect action given a state is violated, since an agent can only see 7x7 squares in front of him. I tried to balance this by
providing the agent with a memory, which he can use to memorize items he picked up as well as where he already was, ..., if he needed it.


## 📊 Results

| Environment Difficulty | Standard PPO Success | **My Curriculum Agent** |
| :--- | :--- | :--- |
| Simple Crossing | 80% | **99%** |
| Memory Maze (Hard) | 12% (Failed) | **92% (Solved)** |

## 🛠️ Tech Stack
* **Language:** Python
* **Library:** PyTorch, Gymnasium (MiniGrid)
* **Algorithm:** Proximal Policy Optimization (PPO) with Recurrent Neural Networks

---
*Created by [Dein Name] - 2024*