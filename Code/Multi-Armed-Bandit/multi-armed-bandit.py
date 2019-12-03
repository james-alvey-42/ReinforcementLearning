import numpy as np
import matplotlib.pyplot as plt

class Machine():
	
	def __init__(self, mean, sigma, label):
		self.mean = mean
		self.sigma = sigma
		self.label = label

	def sample(self):
		return np.random.normal(loc=self.mean, scale=self.sigma, size=1)[0]

class Agent():
	
	def __init__(self, epsilon, Nsteps, Nmachines, label=None):
		self.epsilon = epsilon
		self.Nsteps = Nsteps
		self.label = label
		self.Q = np.zeros((Nmachines, Nsteps))
		self.rewards = np.zeros(Nsteps)
		self.numbers = np.zeros(Nmachines)
		self.pulls = 0
		self.averages = None

	def get_reward(self, machine):
		return machine.sample()

	def pull(self, Q, machines):
		best_machine = np.argmax(Q)
		if (np.random.uniform(0, 1, size=1)[0] > self.epsilon):
			return machines[best_machine]
		else:
			index_sample = np.random.randint(0, len(machines) - 1, size=1)[0]
			while index_sample == best_machine:
				index_sample = np.random.randint(0, len(machines) - 1, size=1)[0]
			return machines[index_sample]

	def update(self, Q, machines):
		pulled_machine = self.pull(Q, machines)
		reward = self.get_reward(pulled_machine)
		num = self.numbers[pulled_machine.label]
		self.numbers[pulled_machine.label] += 1
		Qnew = Q
		Qnew[pulled_machine.label] = (reward + Q[pulled_machine.label]*num)/(num + 1)
		self.pulls += 1
		return Qnew, reward

	def get_averages(self):
		return self.averages

	def get_numbers(self):
		return self.numbers

class ProbabilityAgent():
	
	def __init__(self, Nsteps, alpha, Nmachines, label=None):
		self.Nsteps = Nsteps
		self.label = label
		self.alpha = alpha
		self.H = np.zeros((Nmachines, Nsteps))
		self.probabilities  = (1/Nmachines) * np.ones(Nmachines)
		self.rewards = np.zeros(Nsteps)
		self.numbers = np.zeros(Nmachines)
		self.pulls = 0
		self.averages = np.zeros(Nsteps)

	def get_reward(self, machine):
		return machine.sample()

	def pull(self, machines):
		machine = np.random.choice([i for i in range(len(machines))], 
									size=1,
									p=list(self.probabilities))
		return machines[machine[0]]

	def first_step(self, machines):
		pulled_machine = self.pull(machines)
		reward = pulled_machine.sample()
		self.numbers[pulled_machine.label] += 1
		self.rewards[self.pulls] = reward
		self.averages[0] = reward
		machine_index = pulled_machine.label
		mask = np.ones(len(machines), dtype='bool')
		mask[machine_index] = False
		H = self.H[:, 0]
		H[mask] = H[mask] - self.alpha * (reward - self.averages[0]) * self.probabilities[mask]
		H[machine_index] = H[machine_index] + self.alpha * (reward - self.averages[0]) * (1 - self.probabilities[machine_index])
		self.H[:, 1] = H
		self.probabilities = calculate_probabilities(H)
		self.pulls += 1

	def update(self, machines):
		pulled_machine = self.pull(machines)
		reward = pulled_machine.sample()
		self.numbers[pulled_machine.label] += 1
		self.rewards[self.pulls] = reward
		self.averages[self.pulls] = (reward + self.pulls*self.averages[self.pulls - 1])/(self.pulls + 1)
		machine_index = pulled_machine.label
		mask = np.ones(len(machines), dtype='bool')
		mask[machine_index] = False
		H = self.H[:, self.pulls]
		H[mask] = H[mask] - self.alpha * (reward - self.averages[self.pulls - 1]) *self.probabilities[mask]
		H[machine_index] = H[machine_index] + self.alpha * (reward - self.averages[self.pulls - 1]) * (1 - self.probabilities[machine_index])
		self.H[:, self.pulls + 1] = H
		self.probabilities = calculate_probabilities(H)
		self.pulls += 1

	def get_averages(self):
		return self.averages

	def get_numbers(self):
		return self.numbers

def calculate_probabilities(H):
	denominator = np.sum(np.exp(H))
	numerator = np.exp(H)
	return numerator/denominator

def run_simulation(agent, Nsteps, machines, type='Q'):
	if type == 'Q':
		for i in range(Nsteps - 1):
			Qold = agent.Q[:, i]
			Qnew, reward = agent.update(Qold, machines)
			agent.Q[:, i + 1] = Qnew
			agent.rewards[i] = reward
		averages = []
		for i in range(1, len(agent.rewards)):
			averages.append(np.sum(agent.rewards[:i])*(1/i))
		agent.averages = averages
	elif type == 'H':
		agent.first_step(machines)
		for i in range(Nsteps - 2):
			agent.update(machines)





if __name__ == '__main__':
	np.random.seed(seed=123)

	Nmachines = 10
	means = np.random.uniform(-1, 1, Nmachines)
	sigma = np.random.uniform(0, 1, Nmachines)
	machines = [Machine(mean=means[i], sigma=sigma[i], label=i) for i in range(Nmachines)]
	Nsteps = 10000

	type = 'Q'

	if type == 'Q':

		agent1 = Agent(epsilon=0.1, Nsteps=Nsteps, Nmachines=Nmachines, label='0.1')
		agent2 = Agent(epsilon=0.01, Nsteps=Nsteps, Nmachines=Nmachines, label='0.0')
		agent3 = Agent(epsilon=0.0, Nsteps=Nsteps, Nmachines=Nmachines, label='0.0')

		run_simulation(agent1, Nsteps=Nsteps, machines=machines, type='Q')
		run_simulation(agent2, Nsteps=Nsteps, machines=machines, type='Q')
		run_simulation(agent3, Nsteps=Nsteps, machines=machines, type='Q')


	elif type == 'H':

		agent1 = ProbabilityAgent(alpha=1.0, Nsteps=Nsteps, Nmachines=Nmachines, label='H')
		agent2 = ProbabilityAgent(alpha=0.1, Nsteps=Nsteps, Nmachines=Nmachines, label='H')
		agent3 = ProbabilityAgent(alpha=0.01, Nsteps=Nsteps, Nmachines=Nmachines, label='H')

		run_simulation(agent1, Nsteps=Nsteps, machines=machines, type='H')
		run_simulation(agent2, Nsteps=Nsteps, machines=machines, type='H')
		run_simulation(agent3, Nsteps=Nsteps, machines=machines, type='H')

	# Plotting

	fig = plt.figure(figsize=(18, 18))
	gs = fig.add_gridspec(3, 3)
	
	ax = fig.add_subplot(gs[0:2, 0:2])
	plt.sca(ax)
	if type == 'Q':
		plt.plot(agent1.get_averages(), c='#3AAFA9', label=r'$\epsilon = 0.10$', lw=4.0)
		plt.plot(agent2.get_averages(), c='#F64C72', label=r'$\epsilon = 0.01$', lw=4.0)
		plt.plot(agent3.get_averages(), c='#2F2FA2', label=r'$\epsilon = 0.00$', lw=4.0)
	elif type == 'H':
		plt.plot(agent1.get_averages(), c='#3AAFA9', label=r'$\alpha = 1.00$', lw=4.0)
		plt.plot(agent2.get_averages(), c='#F64C72', label=r'$\alpha = 0.10$', lw=4.0)
		plt.plot(agent3.get_averages(), c='#2F2FA2', label=r'$\alpha = 0.01$', lw=4.0)
	plt.legend(loc='lower right', markerfirst=False, fontsize=24)
	plt.xlabel(r'Number of Steps')
	plt.ylabel(r'Average Reward')
	plt.xscale('log')
	plt.xlim(10,)

	ax = fig.add_subplot(gs[0, 2])
	plt.sca(ax)
	plt.bar([i for i in range(1, Nmachines + 1)], means,
		color='orange',
		edgecolor='orange',
		width=0.4,
		alpha=1.0,
		label='True Mean')
	leg = plt.legend(loc='upper left', frameon=True, handlelength=0, handletextpad=0)
	for item in leg.legendHandles:
		item.set_visible(False)
	plt.xlim(0, Nmachines + 1)
	plt.xticks([i for i in range(1, Nmachines + 1)])

	ax = fig.add_subplot(gs[1, 2])
	plt.sca(ax)
	plt.bar([i for i in range(1, Nmachines + 1)], sigma,
		color='orange',
		edgecolor='orange',
		width=0.4,
		alpha=1.0,
		label='True Variance')
	leg = plt.legend(loc='upper left', frameon=True, handlelength=0, handletextpad=0)
	for item in leg.legendHandles:
		item.set_visible(False)
	plt.xlim(0, Nmachines + 1)
	plt.xticks([i for i in range(1, Nmachines + 1)])
	
	ax = fig.add_subplot(gs[2, 0])
	ax.tick_params(axis='x', which='minor', size=0)
	plt.sca(ax)
	plt.bar([i for i in range(1, Nmachines + 1)], agent1.get_numbers(), 
		color='#3AAFA9',
		edgecolor='#3AAFA9',
		alpha=1.0)
	plt.xlabel('Machine No.')
	plt.ylabel('No. of Pulls')
	plt.yscale('log')
	plt.xlim(0, Nmachines + 1)
	plt.xticks([i for i in range(1, Nmachines + 1)])
	plt.ylim(10, 1.1*Nsteps)
	if type == 'Q':
		plt.text(0.75*Nmachines, 0.5*Nsteps, r'$\epsilon = 0.10$', fontsize=24)
	elif type == 'H':
		plt.text(0.75*Nmachines, 0.5*Nsteps, r'$\alpha = 1.00$', fontsize=24)

	ax = fig.add_subplot(gs[2, 1])
	ax.tick_params(axis='x', which='minor', size=0)
	plt.sca(ax)
	plt.bar([i for i in range(1, Nmachines + 1)], agent2.get_numbers(), 
		color='#F64C72',
		edgecolor='#F64C72',
		alpha=1.0)
	plt.xlabel('Machine No.')
	plt.yscale('log')
	plt.xlim(0, Nmachines + 1)
	plt.xticks([i for i in range(1, Nmachines + 1)])
	plt.ylim(10, 1.1*Nsteps)
	if type == 'Q':
		plt.text(0.75*Nmachines, 0.5*Nsteps, r'$\epsilon = 0.01$', fontsize=24)
	elif type == 'H':
		plt.text(0.75*Nmachines, 0.5*Nsteps, r'$\alpha = 0.10$', fontsize=24)

	ax = fig.add_subplot(gs[2, 2])
	ax.tick_params(axis='x', which='minor', size=0)
	plt.sca(ax)
	plt.bar([i for i in range(1, Nmachines + 1)], agent3.get_numbers(), 
		color='#2F2FA2',
		edgecolor='#2F2FA2',
		alpha=1.0)
	plt.xlabel('Machine No.')
	plt.yscale('log')
	plt.xlim(0, Nmachines + 1)
	plt.xticks([i for i in range(1, Nmachines + 1)])
	plt.ylim(10, 1.1*Nsteps)
	if type == 'Q':
		plt.text(0.75*Nmachines, 0.5*Nsteps, r'$\epsilon = 0.00$', fontsize=24)
	elif type == 'H':
		plt.text(0.75*Nmachines, 0.5*Nsteps, r'$\alpha = 0.01$', fontsize=24)

	fig.suptitle('The Multi Armed Bandit Problem', fontsize=32, fontweight='bold')
	fig.subplots_adjust(top=0.925)
	if type == 'Q':
		plt.savefig('multi-armed-bandit-q.pdf')
	elif type == 'H':
		plt.savefig('multi-armed-bandit-h.pdf')


