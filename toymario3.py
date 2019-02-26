import gym, ppaquette_gym_super_mario
import sys
level="1-1"
env = gym.make('ppaquette/SuperMarioBros-'+level+'-Tiles-v0')

'''
	- First Item -  Up
	- Second Item - Left
	- Third Item -  Down
	- Fourth Item - Right
	- Fifth Item -  A
	- Sixth Item -  B
'''

idle = [0, 0, 0, 0, 0, 0]
right = [0, 0, 0, 1, 0, 0]
jump_right = [0, 0, 0, 1, 1, 0]
left = [0, 1, 0, 0, 0, 0]
jump = [0, 0, 0, 1, 0, 0]


def get_mario_position(state):
	for i in range(state.shape[0]):
		for j in range(state.shape[1]):
			if state[i, j] == 3:
				if state[i,j+1] == 3: #Sometimes mario take two points in state matrix when it is moving. In this case return the point on the right
					return i,j+1
				return i,j
	raise ValueError('mario position not found')

def is_there_cliff(state, columns_considered):
	'''
	Return true if there is cliff within the collumn_considered columns (relative to the current position). 
	'''
	i, j = get_mario_position(state)
	for x_rel in columns_considered:
		if state[-1, min(j+x_rel, state.shape[1]-1)] == 0:
			return True
	return False

def enemy_count(state, columns_considered):
	'''
	Return true if there is enemy within the collumn_considered columns (relative to the current position).
	'''
	i, j = get_mario_position(state)
	count = 0
	for x_rel in columns_considered:
		if j+x_rel < state.shape[1]:
			for y_rel in range(-2, +6):
				y = i + y_rel
				if y >= 0 and y < state.shape[0]:
					if state[y, j+x_rel] == 2:
						if state[y, j+x_rel-1] != 2:    # Sometimes one enemy can appear in two point when it is moving
							count += 1
	return count

def enemy_count_narrow(state, columns_considered):
	'''
	Return true if there is enemy within the collumn_considered columns (relative to the current position).
	'''
	i, j = get_mario_position(state)
	count = 0
	for x_rel in columns_considered:
		if j+x_rel < state.shape[1]:
			for y_rel in range(-1, +1):
				y = i + y_rel
				if y >= 0 and y < state.shape[0]:
					if state[y, j+x_rel] == 2:
						if state[y, j+x_rel-1] != 2:    # Sometimes one enemy can appear in two point when it is moving
							count += 1
	return count

def obstacle_ahead(state):
	i, j = get_mario_position(state)
	if state[i, j+1] == 1:
		return True
	else:
		return False

info = {'distance': 0}
try:
	while info['distance'] != 3252:
		state = env.reset()
		done = False
		i = 0
		old = 40
		reward = 0
		jump_pressed = 0 # In order to release jump_right button for next jump_right
		actions_stored = []
		while not done:
			if actions_stored:
				act = actions_stored[0]
				if act == jump_right:
					jump_pressed +=1
				else:
					jump_pressed = 0
				del actions_stored[0]
			elif i <2 or jump_pressed>8 :
				act = right
				jump_pressed = 0
			else:
				if obstacle_ahead(state):
					act = jump_right
					jump_pressed += 1
				elif is_there_cliff(state, [1]):

					act = jump_right
					jump_pressed += 1

				elif enemy_count_narrow(state, [1,2,3,4, 5])>=2: #and enemy_count(state, [5,6,7])>=1:
					print('@@@@@@@@@@@@@@ Critical moment detected! @@@@@@@@@@@@@@@@@@@@@')
					actions_stored = [jump_right, *([right]*6), *([left]*20), *([right]*6)]
					act = jump_right
					jump_pressed += 1
				elif enemy_count(state, [1,2])>=1:

					act = jump_right
					jump_pressed += 1
				else:
					act = right
					jump_pressed = 0
				#print('---enemy---', enemy_ahead(state, 2))
				#print('---cliff---', cliff_ahead(state, 2))
				#print('---obstacle---', obstacle_ahead(state))
			print('---state--- \n', state)
			print('---reward content--- \n', reward)

			print('---info content--- \n', info)
			input('press enter to continue')
			print('---act---', act)
			s, reward, done, info = env.step(act)
			state = s

			i += 1

			if i % 40 == 0:
				if old == info['distance']:
					actions_stored = [jump_right]*12

			if i % 100 == 0:
				if old == info['distance']:
					break

				else:
					old = info['distance']
		print("Distance: {}".format(info['distance']))
	env.close()
except KeyboardInterrupt:
	env.close()
	exit()
except Exception as e:
	env.close()
	raise e
