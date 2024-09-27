# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import gym  # Reinforcement Learning Environment
import logging  # Logging Library

# ------------------------ Logging Setup ------------------------ #
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

# ------------------------ Part 1: Machine Learning for Emission Prediction ------------------------ #

logging.info("Starting Emission Prediction Model")

try:
    # Load your CSV data
    df = pd.read_csv('cleaned_emissions.csv')
    logging.info("Data loaded successfully")

    # Features: Drop non-numeric or irrelevant columns for ML prediction
    X = df.drop(['year', 'parent_entity', 'parent_type', 'reporting_entity', 
                 'commodity', 'production_unit', 'source', 'total_emissions_MtCO2e'], axis=1)
    y = df['total_emissions_MtCO2e']
    logging.info("Features and target extracted")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets")

    # Initialize and train the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    logging.info("Random Forest Regressor trained successfully")

    # Predict on the test set
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logging.info(f"Mean Absolute Error on Test Set: {mae}")

    # Feature Importance Plot (optional)
    feature_importance = rf_model.feature_importances_
    plt.barh(X.columns, feature_importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Emission Prediction')
    plt.show()

except Exception as e:
    logging.error(f"An error occurred during the ML process: {e}")

# ------------------------ Part 2: Reinforcement Learning for Low-Emission Alternatives ------------------------ #

logging.info("Starting Reinforcement Learning for Emission Alternatives")

# Custom RL Environment for Suggesting Low-Emission Alternatives
class EmissionEnv(gym.Env):
    def __init__(self):
        super(EmissionEnv, self).__init__()
        self.state = [0]  # Placeholder for the initial state (e.g., emission level)
        self.action_space = gym.spaces.Discrete(3)  # 3 alternatives: renewable, efficiency, electrification
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

    def step(self, action):
        emission_reduction = self._get_emission_reduction(action)
        new_state = self.state[0] - emission_reduction
        reward = emission_reduction
        done = new_state <= 0
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

                self.q_table[int(state[0]), action] += self.learning_rate * (
                    reward + self.discount_factor * np.max(self.q_table[int(next_state[0])]) - self.q_table[int(state[0]), action]
                )

                state = next_state

            self.exploration_rate *= self.exploration_decay

# Initialize environment and agent
env = EmissionEnv()
agent = QLearningAgent(env)
logging.info("Q-Learning Agent initialized")

# Train agent
try:
    agent.train(episodes=1000)
    logging.info("Q-Learning Agent training completed")
except Exception as e:
    logging.error(f"An error occurred during RL training: {e}")

# ------------------------ Part 3: Deployable API ------------------------ #

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_emission():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        emission_prediction = rf_model.predict(input_data)[0]

        state = [emission_prediction]
        action = np.argmax(agent.q_table[int(state[0])])
        alternatives = {0: "Switch to Renewable Energy", 1: "Improve Efficiency", 2: "Electrification"}
        recommended_action = alternatives[action]

        return jsonify({"predicted_emission": emission_prediction, "recommended_action": recommended_action})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == '__main__':
    try:
        logging.info("Starting Flask API")
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error starting Flask API: {e}")
