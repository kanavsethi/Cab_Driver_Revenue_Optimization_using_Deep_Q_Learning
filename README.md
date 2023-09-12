# 🚖 Cab Driver Revenue Optimization using Deep Q Learning 📈

Welcome to the Cab Driver Revenue Optimization project! This project aims to develop a reinforcement learning-based algorithm to help cab drivers maximize their profits by making optimal decisions while on the road. With the increasing costs of electricity (all cabs are electric), many drivers have found that their revenues are on the rise, but their profits remain stagnant. The goal is to enable cab drivers to select the 'right' rides, ones that are likely to maximize their total daily profits. 💰🚗

## Problem Statement 🎯

Cab drivers receive ride requests from customers throughout the day. The challenge is to choose the most profitable rides from the available options. Each ride request has its own characteristics, such as distance, location, and potential for additional rides. 🚕📊

## Markov Decision Process (MDP) Formulation 📈

### Objective 🌟

The primary objective of this problem is to maximize the profit earned by the cab driver over the long term.

### Decision Epochs ⏰

Decisions are made at hourly intervals, making the decision epochs discrete.

### Assumptions 🧐

1. The taxis are electric cars that can run for 30 days non-stop (24*30 hours) before needing to recharge. The terminal state is reached when the cab completes 30 days, making an episode 30 days long.
2. There are only 5 locations in the city where the cab can operate.
3. All decisions are made at hourly intervals.
4. Travel times between locations are in integer hours and depend on traffic conditions, which vary by the hour of the day and the day of the week. 🚗🕒

### State 🌐

The state space is defined by the driver's current location, the hour of the day, and the day of the week. A state is represented as (current_location, hour_of_the_day, day_of_the_week), where:

- Number of locations 🏢 (m) = 5
- Number of hours 🕑 (t) = 24
- Number of days 📅 (d) = 7

A terminal state is reached when the cab completes 30 days, making an episode 30 days long.

### Actions 🚀

Every hour, ride requests come from customers in the form of (pick-up, drop) locations. An action is represented by the tuple (pick-up, drop) location. The cab driver can also choose the 'no-ride' action, which advances the time by 1 hour. The number of requests at a state depends on the location, and requests follow a Poisson distribution. 🚕📍

### State Transition 🔄

Given the current state, the next state is determined by the action taken. The time taken to travel from one location to another is retrieved from a precomputed Time Matrix, which depends on traffic conditions and is provided in the project's resources. 🚗🛣️

### Reward 🎁

The objective is to maximize the driver's profit, which is calculated based on the revenue earned from the ride minus the cost of battery consumption during the trip. The reward function is defined as follows:

- Reward at state 𝑠 = 𝑋𝑖 𝑇𝑗 𝐷𝑘:
  - 𝑅(𝑠,𝑎) = 𝑅𝑘 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞)) − 𝐶𝑓 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞) + 𝑇𝑖𝑚𝑒(𝑖, 𝑝)) 𝑎 = (𝑝, 𝑞)
  - 𝑅(𝑠,𝑎) = -𝐶𝑓 𝑎 = (0,0)
 
* Revenue is the amount of money that the cab driver earns by accepting the request
* Cost is the amount of money that the cab driver spends on fuel and other expenses

The cost of fuel is calculated based on the time taken by the cab to the pickup point and then time taken by cab from pickup to the drop point.

#### Time Matrix

The Time Matrix TM.npy represent traffic and is used to calulate time taken by the cab to travel between any two locations.

Time Matrix is a 4-Dimensional matrix. The 4 dimensions are as below:
• Start location 🏢
• End location 🏢
• Time of the day 🕑
• Day of the week 📅

## Solution Approaches 🧠

Two different architectures have been used to solve this problem:

### Architecture 1 🏗️

- Input: Both state and action
- Output: Q Value
- In this approach, to choose maxQ(s′,a), the algorithm calculates Q-values for each possible action and selects the action with the highest Q-value.
- Disadvantage using this Architecture is that it requires a feed forward for each possible action.

### Architecture 2 🏗️

- Input: State only
- Output: Q Values for all actions
- In this approach, the algorithm takes the state as input and directly computes Q-values for all actions. The action with the highest Q-value is chosen.
- Advantage using this Architecture is that it requires a single feed forward for each state action.

## Project Structure 📂

The project's codebase is organized as follows:

- **Time Matrix**: `TM.npy` Time Matrix which represents traffic.
- **Notebooks**: `DQN_Agent_Arch1_state_action_input.ipynb` and `DQN_Agent_Arch2_state_input.ipynb` Agent Code for Architecture 1 and 2 Respectively.
- **H5 and Pickle Files**: Saved Model Weights and Rewards as well as Tracked State Saved objects.
- **Env.py**: Environment containg the rules of the MDP

## Getting Started 🚀

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Explore the provided Jupyter notebooks to understand the data and the implementation of the reinforcement learning algorithms.
3. Run the code and experiments as needed to train and evaluate your model.
4. Modify the code and algorithms as necessary to improve performance.

## Dependencies 📦

This project requires the following dependencies:

- Python 3.x 🐍
- NumPy 🧮
- TensorFlow 🧠
- Matplotlib 📈
- Jupyter Notebook (for experimentation and analysis) 📒

You can install these dependencies using pip or conda.

## Conclusion 🎉

The Cab Driver Revenue Optimization project aims to help cab drivers make more informed decisions to maximize their profits. By formulating the problem as a Markov Decision Process and using Deep Q Learning, this project provides a framework for building and training reinforcement learning models for this specific application.

Feel free to explore, experiment, and improve upon the provided codebase to achieve better results and contribute to the optimization of cab driver revenue. If you have any questions or feedback, please don't hesitate to reach out. 📩

Happy coding and happy driving! 🚗💨👋
