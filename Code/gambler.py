import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_policy(goal):
	policy = {}
	for capital in range(0, goal + 1):
		if capital == 0:
			policy[capital] = 0
		elif capital == goal:
			policy[capital] = 0
		else:
			policy[capital] = np.random.randint(0, capital + 1)
	return policy

def generate_value(goal):
	value = {}
	for capital in range(0, goal + 1):
		if capital == 0:
			value[capital] = 0
		elif capital == goal:
			value[capital] = 1
		else:
			value[capital] = 0
	return value

def print_policy_value(value_dict, policy_dict):
	print("------------------------------------------")
	print("Capital\t\tValue\t\tPolicy Bet")
	print("------------------------------------------")
	for key in value_dict.keys():
		print(key, '\t\t', '{:.3f}'.format(value_dict[key]), '\t\t', '{:.0f}'.format(policy_dict[key]))
	print("------------------------------------------\n")

def plot_policy_values(initial_policy, initial_values, policy_dict, values_dict):
	fig = plt.figure(figsize=(18, 12))

	capital = list(policy_dict.keys())
	policy = list(policy_dict.values())
	values = list(values_dict.values())

	ax = fig.add_subplot(2, 2, 1)
	plt.sca(ax)

	plt.plot(capital, initial_policy, c='#2B4570', label='Random Policy')
	plt.xlabel('Capital')
	plt.ylabel('Policy')
	plt.legend()

	ax = fig.add_subplot(2, 2, 2)
	plt.sca(ax)
	
	plt.plot(capital, policy, c='#F564A9', label='New Policy')
	plt.xlabel('Capital')
	plt.ylabel('Policy')
	plt.legend()

	ax = fig.add_subplot(2, 2, 3)
	plt.sca(ax)
	
	plt.plot(capital, initial_values, c='#2B4570', label='Initial Value')
	plt.xlabel('Capital')
	plt.ylabel('Value')
	plt.legend()

	ax = fig.add_subplot(2, 2, 4)
	plt.sca(ax)
	
	plt.plot(capital, values, c='#F564A9', label='New Value')
	plt.xlabel('Capital')
	plt.ylabel('Value')
	plt.legend()

	plt.savefig('Figures/policy.pdf')


class Gambler():

	def __init__(self, goal, gamma, ph, policy=None):
		self.goal = goal
		self.discount = gamma
		self.ph = ph
		if policy == None:
			self.policy = generate_policy(goal)
		else:
			self.policy = policy
		self.value = generate_value(goal)

	def evaluate_policy(self):
		for s in range(1, self.goal):
			for sp in range(0, s):
				bet = sp
				reward = np.heaviside(sp + s - self.goal, 1)
				values = []
				if s + bet > self.goal:
					values.append(self.ph * (reward + self.discount * self.value[self.goal]) + (1 - self.ph) * self.discount * self.value[s - bet])
				else:
					values.append(self.ph * (reward + self.discount * self.value[s + bet]) + (1 - self.ph) * self.discount * self.value[s - bet])
			self.value[s] = np.max(values)

	def update_policy(self):
		for s in range(1, self.goal):
			policies = []
			for sp in range(0, s):
				reward = np.heaviside(sp + s - self.goal, 1)
				if s + sp > self.goal:
					policies.append(self.ph * (reward + self.discount * self.value[self.goal]) + (1 - self.ph) * self.discount * self.value[s - sp])
				else:
					policies.append(self.ph * (reward + self.discount * self.value[s + sp]) + (1 - self.ph) * self.discount * self.value[s - sp])
			policies = np.array(policies)
			self.policy[s] = np.argmax(policies)




if __name__ == '__main__':
	goal = 100
	gamma = 0.9
	ph = 0.1
	initial_policy = generate_policy(goal)
	gambler = Gambler(goal=goal, 
		gamma=gamma, 
		ph=ph, 
		policy=initial_policy)

	initial_policy = list(initial_policy.values())
	initial_values = list(gambler.value.values())

	#print_policy_value(gambler.value, gambler.policy)

	policy_iters = 1000
	value_iters = 10

	for _ in range(policy_iters):
		print('Completed {} out of {}'.format(_ + 1, policy_iters), end='\r')

		for _ in range(value_iters):
			gambler.evaluate_policy()

		gambler.update_policy()

	#print_policy_value(gambler.value, gambler.policy)

	plot_policy_values(initial_policy, initial_values, gambler.policy, gambler.value)
		
		
