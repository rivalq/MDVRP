from functools import lru_cache
from math import sqrt
from TSP import *
import random


# n + m  nodes, m depots (n+1...n+m)


n = 50
m = 4

##input n + m lines (xi,yi)

x = [37,49,52,20,40,21,17,31,52,51,42,31,5,12,36,52,37,49,27,17,13,57,62,42,16,8,7,27,30,43,58,58,37,38,46,61,32,45,59,5,10,21,5,30,39,32,25,25,48,56,20,30,50,60]
y = [52,49,64,26,30,47,63,62,33,21,41,32,25,42,16,41,23,33,41,13,58,42,57,57,52,38,68,67,48,27,69,46,10,33,63,69,22,35,15,6,17,10,64,15,10,39,32,55,28,37,20,40,30,50]


"""
for i in range(n+m):
	arr = input().split(" ")
	x.append(int(arr[0]))
	y.append(int(arr[1]))


"""

def func(i,j):
	dx = (x[i] - x[j])**2
	dy = (y[i] - y[j])**2

	return sqrt(dx + dy)


dis = []

for i in range(0,n+m):
	dis.append([])
	for j in range(0,n + m):
		dis[i].append(func(i,j))


"""
n = 16

m = 4
dis = [[0, 10, 20, 10, 22, 20, 22, 28, 14, 14, 31, 14, 31, 31, 31, 42, 28, 22, 44, 22],
[10, 0, 10, 14, 14, 22, 20, 22, 22, 10, 22, 22, 22, 36, 30, 36, 36, 20, 36, 31],
[20, 10, 0, 22, 10, 28, 22, 20, 31, 14, 14, 31, 14, 42, 31, 31, 44, 22, 28, 41],
[10, 14, 22, 0, 20, 10, 14, 22, 22, 22, 36, 10, 30, 22, 22, 36, 36, 31, 50, 20],
[22, 14, 10, 20, 0, 22, 14, 10, 36, 22, 22, 30, 10, 36, 22, 22, 50, 31, 36, 40],
[20, 22, 28, 10, 22, 0, 10, 20, 31, 31, 42, 14, 31, 14, 14, 31, 44, 41, 56, 22],
[22, 20, 22, 14, 14, 10, 0, 10, 36, 30, 36, 22, 22, 22, 10, 22, 50, 40, 50, 31],
[28, 22, 20, 22, 10, 20, 10, 0, 42, 31, 31, 31, 14, 31, 14, 14, 56, 41, 44, 41],
[14, 22, 31, 22, 36, 31, 36, 42, 0, 20, 40, 20, 44, 40, 44, 56, 14, 22, 50, 22],
[14, 10, 14, 22, 22, 31, 30, 31, 20, 0, 20, 28, 28, 44, 40, 44, 31, 10, 31, 36],
[31, 22, 14, 36, 22, 42, 36, 31, 40, 20, 0, 44, 20, 56, 44, 40, 50, 22, 14, 53],
[14, 22, 31, 10, 30, 14, 22, 31, 20, 28, 44, 0, 40, 20, 28, 44, 31, 36, 58, 10],
[31, 22, 14, 30, 10, 31, 22, 14, 44, 28, 20, 40, 0, 44, 28, 20, 58, 36, 31, 50],
[31, 36, 42, 22, 36, 14, 22, 31, 40, 44, 56, 20, 44, 0, 20, 40, 50, 53, 70, 22],
[31, 30, 31, 22, 22, 14, 10, 14, 44, 40, 44, 28, 28, 20, 0, 20, 58, 50, 58, 36],
[42, 36, 31, 36, 22, 31, 22, 14, 56, 44, 40, 44, 20, 40, 20, 0, 70, 53, 50, 53],
[28, 36, 44, 36, 50, 44, 50, 56, 14, 31, 50, 31, 58, 50, 58, 70, 0, 30, 60, 30],
[22, 20, 22, 31, 31, 41, 40, 41, 22, 10, 22, 36, 36, 53, 50, 53, 30, 0, 30, 42],
[44, 36, 28, 50, 36, 56, 50, 44, 50, 31, 14, 58, 31, 70, 58, 50, 60, 30, 0, 67],
[22, 31, 41, 20, 40, 22, 31, 41, 22, 36, 53, 10, 50, 22, 36, 53, 30, 42, 67, 0]
]
"""

max_capacity = 5000

demand = []

for i in range(0,n):
	demand.append(random.choice(range(100,501)))

for i in range(n,n+m):
	demand.append(0)	


max_time = 2
avg_speed = 40

max_distance = max_time*avg_speed

INF = 10**18

magic = 10  ## maximum limit to run naive algorithm

bucket_size = 10


max_routes = 8
best_cost = INF
new_cost = INF

max_iterations = 10
max_no_improve = 2

class route:
	def __init__(self,depot,path):
		self.depot = depot
		self.path = path

	def print_path(self):
		new_path = []
		for i in range(len(self.path)):
			new_path.append(self.path[i])
		new_path.append(self.depot)	
		print(new_path)				

## we can represent solution as set of routes

## if we didnt add nodes to a previous route we will take that route as it is
## other wise take the new route

## in normal form dp will be  [route_index/depot_index,mask]

## to set contraints on number of vehicles we have to add new dimension in dp [num of routes]

## complexity of dp will be (3^p)*(number of routes)





class parent:
	def __init__(self,mask,j,value):
		self.mask = mask
		self.j = j
		self.value = value
	def relax(self,mask,j,value):
		if(self.value > value):
			self.value = value
			self.mask = mask
			self.j = j	
class model:

	def __init__(self):
		pass	
	



	@lru_cache(maxsize = None)	
	def cost(self,nodes):
		set_nodes = set(nodes)
		

		_cap_ = 0
		for i in set_nodes:
			_cap_ += demand[i]

		if(_cap_ > max_capacity):
			return INF	

		set_nodes = list(set_nodes)
		
		num_nodes = len(set_nodes)

		dis_tsp = []

		for i in range(num_nodes):
			dis_tsp.append([])
			for j in range(num_nodes):
				dis_tsp[i].append(dis[set_nodes[i]][set_nodes[j]])


		tsp = TSP(dis_tsp)
		tsp.MST()
		path = tsp.Build()
		_cost_ = 0
		_dis_ = 0	

		depot_index = -1

		for i in range(len(path)):
			path[i] = set_nodes[path[i]]
			if(path[i] >= n):
				depot_index = i

		shifted_path = []
		for i in range(len(path)):
			shifted_path.append(path[(i + depot_index)%len(path)])		

		path = shifted_path
			
		dis1 = dis[path[0]][path[1]]
		dis2 = dis[path[0]][path[-1]]

		if(dis1 > dis2):
			temp = []
			for i in range(1,len(path)):
				temp.append(path[i])
			temp = temp[::-1]
			path = [path[0]]
			for i in temp:
				path.append(i)		




		for i in range(len(path)):
			j  = (i - 1)%len(path)
			_cost_ += dis[path[i]][path[j]]
			
		_dis_ = _cost_ - max(dis1,dis2)

		if(_dis_ > max_distance):
			return INF	

		return _cost_		


	def optimal_route(self,nodes):
		set_nodes = set(nodes)
		
		set_nodes = list(set_nodes)
		
		num_nodes = len(set_nodes)

		dis_tsp = []

		for i in range(num_nodes):
			dis_tsp.append([])
			for j in range(num_nodes):
				dis_tsp[i].append(dis[set_nodes[i]][set_nodes[j]])



		tsp = TSP(dis_tsp)
		tsp.MST()
		path = tsp.Build()
		depot_index = -1

		for i in range(len(path)):
			path[i] = set_nodes[path[i]]
			if(path[i] >= n):
				depot_index = i
		shifted_path = []
		for i in range(len(path)):
			shifted_path.append(path[(i + depot_index)%len(path)])		

		path = shifted_path
			


		dis1 = dis[path[0]][path[1]]
		dis2 = dis[path[0]][path[-1]]

		if(dis1 > dis2):
			temp = []
			for i in range(1,len(path)):
				temp.append(path[i])
			temp = temp[::-1]
			path = [path[0]]
			for i in temp:
				path.append(i)		

		depot = path[0]			

		return route(depot,path)	


	def solve(self,bucket,routes,result):
		# bucket is the list of nodes to be added to initial solution

		## for each subset we will either create a new route or add into existing route
		## to get best route we will solve the tsp problem

		p = len(bucket)

		

		num_routes = len(routes)
		


		k = m + num_routes +1		
		dp = []
		for i in range(k):
			dp.append([])
			for j in range(1 << p):
				dp[i].append([])
				for _ in range(max_routes+1):
					dp[i][j].append(INF)	


		
		par = []			
		


		for i in range(k):
			par.append([])
			for j in range(1 << p):
				par[i].append([])
				for _ in range(max_routes+1):
					par[i][j].append(parent(0,0,INF))



		dp[0][0][0] = 0
		for _ in range(0,max_routes+1):
			dp[0][0][_] = 0	
		## first part of dp


		for i  in range(1,m+1):
			for _ in range(0,max_routes+1):
				dp[i][0][_] = 0
				par[i][0][_] = parent(0,_,0)
			for mask in range(1,1 << p):

				for j in range(1,max_routes+1): 	

					

					# Either we will create a new route with depot # n+i-1
					# Either we will do nothing
					# 1. dp[i][mask][j] = min(dp[i-1][mask^submask][j-1] + cost_new_route)
					# 2. dp[i][mask][j] = dp[i-1][mask][j]


					submask = mask
					dp[i][mask][j] = dp[i-1][mask][j]

					par[i][mask][j].relax(mask,j,dp[i-1][mask][j])
					while(submask > 0):
						nodes = [n + i - 1]
						for element in range(0,p):
							if((1 << element) & submask):
								nodes.append(bucket[element])
						_cost_  = self.cost(tuple(nodes))



						par[i][mask][j].relax(mask^submask,j-1,dp[i-1][mask^submask][j-1]+_cost_)						
						dp[i][mask][j] = min(dp[i][mask][j],dp[i-1][mask^submask][j-1] + _cost_)
						submask = (submask-1)&mask



		## second part of dp				

		for i in range(m+1,k):
			
			for mask in range(0, 1 << p):

				for j in range(1,max_routes+1):


					## we have two option
					## 1. Merge existing route with some submask of mask
					## 2. Add this route as it is

					__cost__ = self.cost(tuple(routes[i - m - 1]))
					dp[i][mask][j] = dp[i - 1][mask][j-1] + __cost__
					par[i][mask][j].relax(mask,j-1,dp[i - 1][mask][j-1]+__cost__)

					submask = mask

					while(submask):
						nodes = routes[i - m - 1][:]
						for element in range(0,p):
							if((1 << element) & submask):
								nodes.append(bucket[element])
						_cost_  = self.cost(tuple(nodes))

						par[i][mask][j].relax(mask^submask,j-1,dp[i-1][mask^submask][j-1]+_cost_)						
						dp[i][mask][j] = min(dp[i][mask][j],dp[i-1][mask^submask][j-1] + _cost_)

						submask = (submask-1)&mask		


		## Complexity - O(3^p*(m+num_routes)*max_routes) ~ O((3^p)*(max_routes^2))					

		## now we need to take minimum of dp[k-1][(1<<p)-1][i], 1 <= i <= max_routes
		

		final_state = parent(0,0,2*INF)

		for i in range(1,max_routes+1):
			final_state.relax((1<<p)-1,i,dp[k-1][(1<<p)-1][i])


		result[0] = final_state.value
			

		new_routes = []

		## backtrack part

		mask = final_state.mask
		i = k - 1
		j = final_state.j


		while(i > 0):
			state = par[i][mask][j]
			if(i > m):
				diff = mask^state.mask
				if(diff == 0):
					optimal_r = self.optimal_route(tuple(routes[i - m - 1]))
					new_routes.append(optimal_r)
					## same as previous route
				else:
					nodes = routes[i - m - 1][:]
					for j in range(0,p):
						if((1<<j) & diff):
							nodes.append(bucket[j])
					optimal_r = self.optimal_route(nodes)
					new_routes.append(optimal_r)		
					## merged route	

				mask = state.mask	
				i -= 1
				j = state.j
			else:
				diff = mask^state.mask
				if(diff == 0):
					pass
					## no route added

				else:
					nodes = [i + n - 1]
					
					for j in range(0,p):
						if((1<<j) & diff):
							nodes.append(bucket[j])
					optimal_r = self.optimal_route(nodes)
					new_routes.append(optimal_r)	
					## new route added with depot i + n -1		

				mask = state.mask
				i -= 1
				j = state.j	

		return new_routes	

	def inital(self,result):


		## if n < 10 directly create the best solution
		## else create buckets of size 10 


		if(n <= magic):
			bucket = []
			routes = []
			for i in range(0,n):
				bucket.append(i)
			new_routes = self.solve(bucket,routes,result)
			return new_routes	

		else:
			bucket = []
			routes = []
			new_routes = []
			for i in range(0,n,bucket_size):
				t = []
				for j in range(i,min(n,i+bucket_size)):
					t.append(j)
				bucket.append(t)
			new_routes = self.solve(bucket[0],routes,result)	
			
			for i in range(1,len(bucket)):
				routes = []
				for j in new_routes:
					routes.append(j.path)
				
				new_routes = self.solve(bucket[i],routes,result)
				
			return new_routes	


mo = model()
temp = [best_cost]

routes = mo.inital(temp)

best_cost = temp[0]

print(best_cost)

for i in routes:
	i.print_path()


class ALNS:
	
	def __init__(self):
		
		self.prob  = []

		for i in range(4):
			self.prob.append(0.25)
		self.delta = 0.01

	def destroy(self,bucket):
		new_routes = []
		for i in routes:
			r = []
			for j in i.path:
				if(j not in bucket):
					r.append(j)
			if(len(r) > 1):
				new_routes.append(r)
		return new_routes					
		

	def operator_1(self):
		global best_cost,routes

		bucket = random.sample(range(0,n),bucket_size)
		old_routes = self.destroy(bucket)
		_cost_ = [INF]
		new_routes = mo.solve(bucket,old_routes,_cost_)
		

		if(_cost_[0] > best_cost):
			return 0
		else:
			best_cost = _cost_[0]
			routes = new_routes
			return 1


	def nearest(self,u,k):
		nearest_dis = []
		for i in range(0,n):
			if(i != u):
				nearest_dis.append([i,dis[u][i]])
		nearest_dis.sort(key = lambda e: e[1])		
		res = []

		for i in range(k):
			res.append(nearest_dis[i][0])
		return res		

	def operator_2(self):
		global best_cost,routes

		head = random.sample(range(0,n),1)[0]
		nearest_dis = self.nearest(head,bucket_size-1)
		bucket = nearest_dis
		bucket.append(head)

		old_routes = self.destroy(bucket)
		_cost_ = [INF]
		new_routes = mo.solve(bucket,old_routes,_cost_)
		if(_cost_[0] > best_cost):
			return 0
		else:
			best_cost = _cost_[0]
			routes = new_routes
			return 1
		

	def operator_3(self):
		global best_cost,routes
		if(bucket_size % 2):
			print("error","bucket_size should be even")	
			exit(0)

		heads = random.sample(range(0,n),bucket_size//2)
		nodes = []
		for i in range(len(heads)):
			nearest_dis = self.nearest(heads[i],n-1)
			nodes.append(heads[i])
			for j in nearest_dis:
				if((j not in heads) and (j not in nodes)):
					nodes.append(j)
					break
		bucket = nodes	
		old_routes = self.destroy(bucket)
		_cost_ = [INF]
		new_routes = mo.solve(bucket,old_routes,_cost_)
		if(_cost_[0] > best_cost):
			return 0
		else:
			best_cost = _cost_[0]
			routes = new_routes
			return 1		

	def operator_4(self):

		global best_cost,routes

		num_routes = len(routes)
		routes_id = []
		for i in range(num_routes):
			routes_id.append(i)	
		random.shuffle(routes_id)

		bucket = []

		for i in routes_id:
			if(len(bucket) == bucket_size):
				break
			temp = routes[i].path
			for j in temp:
				if(bucket_size == len(bucket)):
					break
				if(j < n):
					bucket.append(j)

		old_routes = self.destroy(bucket)
		_cost_ = [INF]
		new_routes = mo.solve(bucket,old_routes,_cost_)
		if(_cost_[0] > best_cost):
			return 0
		else:
			best_cost = _cost_[0]
			routes = new_routes
			return 1				


	def run_algo(self):
		itr = 0
		for _ in range(max_iterations):
			found_best = 0
			operator_id = random.choices([1,2,3,4], weights = self.prob,k =1)[0]

			if(operator_id == 1):

				for j in range(max_no_improve):
					found_best = self.operator_1()
					if(found_best == 1):
						break
				sum_p = 0
				for j in range(0,4):
					sum_p += self.prob[j]		
				if(found_best == 0):
					self.prob[0] -=  self.delta
				else:
					self.prob[0] += self.delta/sum_p

			elif(operator_id == 2):

				for j in range(max_no_improve):
					found_best = self.operator_2()
					if(found_best == 1):
						break
				sum_p = 0
				for j in range(0,4):
					sum_p += self.prob[j]		
				if(found_best == 0):
					self.prob[1] -=  self.delta
				else:
					self.prob[1] += self.delta/sum_p

			elif(operator_id == 3):

				for j in range(max_no_improve):
					found_best = self.operator_3()
					if(found_best == 1):
						break
				sum_p = 0
				for j in range(0,4):
					sum_p += self.prob[j]		
				if(found_best == 0):
					self.prob[2] -=  self.delta
				else:
					self.prob[2] += self.delta/sum_p

			elif(operator_id == 4):

				for j in range(max_no_improve):
					found_best = self.operator_4()
					if(found_best == 1):
						break
				sum_p = 0
				for j in range(0,4):
					sum_p += self.prob[j]		
				if(found_best == 0):
					self.prob[3] -=  self.delta
				else:
					self.prob[3] += self.delta/sum_p

			itr += 1

			print("solution after {0} iteratios".format(itr))

			print("cost = {0}".format(best_cost))

			print("routes")

			for i in routes:
				i.print_path()		

											

alns = ALNS()

alns.run_algo()

print(best_cost)

for i in routes:
	i.print_path()
