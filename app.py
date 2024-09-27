# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import gym  # Reinforcement Learning Environment
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load your CSV data
df = pd.read_csv('cleaned_emissions.csv')

# Features: Drop non-numeric or irrelevant columns for ML prediction
X = df.drop(['year', 'parent_entity', 'parent_type', 'reporting_entity', 
             'commodity', 'production_unit', 'source', 'total_emissions_MtCO2e'], axis=1)

# Target: Total Emissions (the variable we want to predict)
y = df['total_emissions_MtCO2e']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on Test Set: {mae}")

# Feature Importance Plot (optional)
feature_importance = rf_model.feature_importances_
plt.barh(X.columns, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Emission Prediction')
plt.show()

# Custom RL Environment for Suggesting Low-Emission Alternatives
class EmissionEnv(gym.Env):
    def __init__(self):
        super(EmissionEnv, self).__init__()
        self.state = [0]  # Placeholder for the initial state (e.g., emission level)
        self.action_space = gym.spaces.Discrete(3)  # 3 alternatives
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        
    def step(self, action):
        emission_reduction = self._get_emission_reduction(action)
        new_state = self.state[0] - emission_reduction  # Reduce emission based on action taken
        reward = emission_reduction  # Reward is the emission reduction
        done = new_state <= 0  # Done when emission is 0 or less
        self.state = [new_state]
        return np.array(self.state), reward, done, {}
    
    def reset(self):
        self.state = [np.random.uniform(50, 100)]  # Random initial emission level
        return np.array(self.state)
    
    def _get_emission_reduction(self, action):
        if action == 0:  # Renewable energy
            return np.random.uniform(15, 25)
        elif action == 1:  # Efficiency improvement
            return np.random.uniform(10, 20)
        elif action == 2:  # Electrification
            return np.random.uniform(20, 30)

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.q_table = np.zeros((100, env.action_space.n))  # Q-table for state-action values
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                if np.random.uniform(0, 1) < self.exploration_rate:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[int(state[0])])
                
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-value
                self.q_table[int(state[0]), action] = (self.q_table[int(state[0]), action] +
                                                       self.learning_rate * (reward + 
                                                       self.discount_factor * np.max(self.q_table[int(next_state[0])]) - 
                                                       self.q_table[int(state[0]), action]))
                
                state = next_state
                
            # Reduce exploration over time
            self.exploration_rate *= self.exploration_decay

# Initialize environment and agent
env = EmissionEnv()
agent = QLearningAgent(env)

# Train agent
agent.train(episodes=1000)

# Pydantic model for input validation
class EmissionInput(BaseModel):
    feature1: float  # Replace with your actual feature names and types
    feature2: float  # Add more features as needed

# Endpoint to predict emission and recommend an alternative
@app.post('/predict')
def predict_emission(input_data: EmissionInput):
    input_df = pd.DataFrame([input_data.dict()])
    emission_prediction = rf_model.predict(input_df)[0]
    state = [emission_prediction]
    action = np.argmax(agent.q_table[int(state[0])])
    alternatives = {0: "Switch to Renewable Energy", 1: "Improve Efficiency", 2: "Electrification"}
    recommended_action = alternatives[action]
    return {"predicted_emission": emission_prediction, "recommended_action": recommended_action}

if __name__ == '__main__':
    app.run(debug=True)
