import csv
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg.linalg import _is_empty_2d


data = ['generations', 'pc', 'pm', 'Q', 'population_size',
        'itermax', 'numant', 'shortest', 'round(end-start,1)']
f = open('aco_ga_itermax.csv', 'a', newline='')
csv_write = csv.writer(f, dialect='excel')
csv_write.writerow(data)
f.close()

coordinates = np.array([[37.0, 52.0], [49.0, 49.0], [52.0, 64.0], [20.0, 26.0], [40.0, 30.0],
                        [21.0, 47.0], [17.0, 63.0], [31.0, 62.0], [
                            52.0, 33.0], [51.0, 21.0],
                        [42.0, 41.0], [31.0, 32.0], [5.0, 25.0], [
                            12.0, 42.0], [36.0, 16.0],
                        [52.0, 41.0], [27.0, 23.0], [17.0, 33.0], [
                            13.0, 13.0], [57.0, 58.0],
                        [62.0, 42.0], [42.0, 57.0], [
                            16.0, 57.0], [8.0, 52.0], [7.0, 38.0],
                        [27.0, 68.0], [30.0, 48.0], [43.0, 67.0], [
                            58.0, 48.0], [58.0, 27.0],
                        [37.0, 69.0], [38.0, 46.0], [46.0, 10.0], [
                            61.0, 33.0], [62.0, 63.0],
                        [63.0, 69.0], [32.0, 22.0], [
                            45.0, 35.0], [59.0, 15.0], [5.0, 6.0],
                        [10.0, 17.0], [21.0, 10.0], [5.0, 64.0], [
                            30.0, 15.0], [39.0, 10.0],
                        [32.0, 39.0], [25.0, 32.0], [25.0, 55.0], [
                            48.0, 28.0], [56.0, 37.0],
                        [30.0, 40.0]])
# shape[0]=52 城市个数,也就是任务个数
# 生成初始种群 α∈（0,3） β∈（2,5） ρ∈（0,0.5）
numcity = coordinates.shape[0]


def getdistmat():
    # num = coordinates.shape[0]
    dist = np.zeros((numcity, numcity))
    # 初始化生成52*52的矩阵
    for i in range(numcity):
        for j in range(i, numcity):
            dist[i, j] = dist[j, i] = np.linalg.norm(
                coordinates[i] - coordinates[j])
            # dist[i][j] = dist[j][i] = round(ne.evaluate("((
            # coordinates[i] - coordinates[j])**2)**0.5"),1)
    return dist


distmat = getdistmat()


def init_population():
    global population
    N = np.random.uniform(size=(POP_SIZE, 4))  # 做一个5X4的随机矩阵
    for i in range(POP_SIZE):
        N[i, 0] = i + 1  # 序号列
        N[i, 1] = np.random.uniform(0, 3)  # α∈（0,3）信息素权重因子
        N[i, 2] = np.random.uniform(2, 5)  # β∈（2，5）启发函数重要程度因子
        N[i, 3] = np.random.uniform(0, 0.5)  # ρ∈（0,0.5) 信息素的挥发速度

        population = N
     
    # return population  # # # 


def aco():
    global numcity, coordinates
    etaTable = 1.0 / (distmat + np.diag([1e10] * numcity))
    # η  diag(),将一维数组转化为方阵 启发函数矩阵，表示蚂蚁从城市i转移到城市j的期望程度
    pheromoneTable = np.ones((numcity, numcity))  # 信息素矩阵 52*52
    pathTable = np.zeros((NUMANT, numcity)).astype(int)  # 路径记录表，转化成整型 40*52

    lenAver = np.zeros(itermax)  # 迭代50次，存放每次迭代后，路径的平均长度  50*1
    lenBest = np.zeros(itermax)  # 迭代50次，存放每次迭代后，最佳路径长度  50*1
    pathBest = np.zeros((itermax, numcity))  # 迭代50次，存放每次迭代后，最佳路径城市的坐标 50*52
    for q in population:  # 五组参数在蚁群算法中循环
        alpha = q[1]
        beta = q[2]
        rho = q[3]
        for iter in range(itermax):
            # 迭代总数
            # 40个蚂蚁随机放置于52个城市中
            if NUMANT <= numcity:  # 城市数比蚂蚁数多，不用管
                pathTable[:, 0] = np.random.permutation(range(numcity))[
                    :NUMANT]
                # 返回一个打乱的40*52矩阵，但是并不改变原来的数组,把这个数组的第一列(40个元素)放到路径表的第一列中
                # 矩阵的意思是哪个蚂蚁在哪个城市,矩阵元素不大于52
            else:  # 蚂蚁数比城市数多，需要有城市放多个蚂蚁
                pathTable[:numcity, 0] = np.random.permutation(range(numcity))[
                    :]
                # 先放52个
                pathTable[numcity:, 0] = np.random.permutation(range(numcity))[
                    :NUMANT - numcity]
                # 再把剩下的放完
            # print(pathTable[:,0])
            length = np.zeros(NUMANT)  # 1*40的数组
            # 本段程序算出每只/第i只蚂蚁转移到下一个城市的概率
            for i in range(NUMANT):
                # i=0
                visiting = pathTable[i, 0]  # 当前所在的城市
                # set()创建一个无序不重复元素集合
                # visited = set() #已访问过的城市，防止重复
                # visited.add(visiting) #增加元素
                # print(visited)
                unvisited = set(range(numcity))  # 未访问的城市集合                
                unvisited.remove(visiting)  # 删除已经访问过的城市元素
                for j in range(1, numcity):  # 循环numcity-1次，访问剩余的所有numcity-1个城市
                    # 每次用轮盘法选择下一个要访问的城市
                    listUnvisited = list(unvisited)  # 未访问城列表                  
                    probTrans = np.zeros(len(listUnvisited))
                    # 每次循环都初始化转移概率矩阵1*52,1*51,1*50,1*49....
                    
                    # 以下是计算转移概率
                    # for k in range(len(listUnvisited)):
                    probTrans[:] = (pheromoneTable[visiting, listUnvisited[:]]**alpha) * (etaTable[visiting, listUnvisited[:]]**beta)
                    
                    # eta-从城市i到城市j的启发因子 这是概率公式的分母
                    # 其中[visiting][listunvis[k]]是从本城市到k城市的信息素
                    cumsumProbTrans = (probTrans / sum(probTrans)).cumsum()-np.random.rand()
                    # 求出本只蚂蚁的转移到各个城市的概率斐波衲挈数列 随机生成下个城市的转移概率，再用区间比


                    # k = listunvisited[ndarray.find(cumsumprobtrans > 0)[0]]
                    k = listUnvisited[list(cumsumProbTrans > 0).index(True)]
                    # k = listunvisited[np.where(cumsumprobtrans > 0)[0]]
                    # where 函数选出符合cumsumprobtans>0的数
                    # 下一个要访问的城市
                    pathTable[i, j] = k
                    # print(pathTable)
                    # 采用禁忌表来记录蚂蚁i当前走过的第j城市的坐标，这里走了第j个城市.k是中间值
                    unvisited.remove(k)  # 将未访问城市列表中的K城市删去
                    length[i] += distmat[visiting, k]
                    # 计算本城市到K城市的距离
                    visiting = k
                length[i] += distmat[visiting, pathTable[i, 0]]
                # 计算本只蚂蚁的总的路径距离，包括最后一个城市和第一个城市的距离
            # print("ants all length:",length)
            # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数（每只蚂蚁遍历完城市后总路程）
            lenAver[iter] = length.mean()
            # 本轮的平均路径
            # 本部分是为了求出最佳路径 更新所有的蚂蚁
            if iter == 0:
                lenBest[iter] = length.min()
                pathBest[iter] = pathTable[length.argmin()].copy()
            # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
            else:
                # 后面几轮的情况，更新最佳路径
                if length.min() > lenBest[iter - 1]:
                    lenBest[iter] = lenBest[iter - 1]
                    pathBest[iter] = pathBest[iter - 1].copy()
                # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
                else:
                    lenBest[iter] = length.min()
                    pathBest[iter] = pathTable[length.argmin()].copy()
            # 此部分是为了更新信息素
            chgPheromoneTable = np.zeros((numcity, numcity))
            for i in range(NUMANT):  # 更新所有的蚂蚁
                for j in range(numcity - 1):
                    chgPheromoneTable[pathTable[i, j]
                            , pathTable[i, j + 1]] += Q / distmat[pathTable[i, j], pathTable[i, j + 1]]
                    # 根据公式更新本只蚂蚁改变的城市间的信息素      Q/d   其中d是从第j个城市到第j+1个城市的距离
                chgPheromoneTable[pathTable[i, j + 1]
                            , pathTable[i, 0]] += Q / distmat[pathTable[i, j + 1], pathTable[i, 0]]
                # 首城市到最后一个城市 所有蚂蚁改变的信息素总和

            # 信息素更新公式p=(1-挥发速率)*现有信息素+改变的信息素
            pheromoneTable = (1 - rho) * pheromoneTable + chgPheromoneTable
            # iter += 1  # 迭代次数指示器+1
            # print("this iteration end：", iter)
            # 观察程序执行进度，该功能是非必须的
            # if (iter - 1) % 20 == 0:
            #    print("schedule:",iter - 1)
        # 迭代完成
        path = np.array(pathBest[-1])
        shortLen = lenBest[iter - 1]  # 取出最短路径
        # print('当前参数最短路径:',shortlength)          #输出每组参数的最短路径
        global shortest
        shortest = np.append(shortest, shortLen)
        # np.delete(lengthbest,0,axis=0)
        popEachList = [alpha, beta, rho, shortLen, path]
        popAllList.append(popEachList)
        # popAllList = np.concatenate(popAllList, popEachList)
    #print('返回所有参数组合所计算的最短路径', shortest)
    # print("种群表", population_alllist)  # 制作一个种群表[first[alpha,beta,rho,lengthbest,[pathbest]],second[...]...]避免对应错误

# 适应值评价


def calculate_fitness():
    # for i in range(POP_SIZE):
    #     function_value = 1/popAllList[i][3]
    #     fitness.append(function_value)
    global fitness
    fitness = (1/np.array(popAllList, dtype=object)[:,3]).tolist()
    # print('每个种群的适应值：',fitness)

# 获取最大适应度的个体(每轮遗传最优)


def best_value():
    '''
    找到fitness中最大的那个的序号赋给best_fitness
    '''
    # best_fitness = popAllList[fitness.index(max(fitness))]
    # min_fitness = popAllList[0][3]
    # for pop in popAllList[:][3]:
    #     if pop < min_fitness:
    #         min_fitness = pop[3]
    #         best_fitness = pop
    # optimum_solution.append(best_fitness)
    optimum_solution.append(popAllList[fitness.index(max(fitness))])
    # print('遗传迭代每轮最优种群表：',optimum_solution)
    # return optimum_solution

# 获取所有遗传最优


def best_ga():
    '''
    找到optimum_solution里最小的那个
    '''
    # ga_best = optimum_solution[optimum_solution.index(min(np.array(optimum_solution, dtype=object)[:,3]))]
    # ga_best = optimum_solution[0]
    # ga_good = optimum_solution[0][3]
    # for i in range(GENERATIONS):
    #     if optimum_solution[i][3] < ga_good:
    #         ga_good = optimum_solution[i][3]
    #         ga_best = optimum_solution[i]
    # best_solution.extend(ga_best)
    # global best_solution
    opt_s3 = np.array(optimum_solution, dtype=object)[:,3]
    best_solution.extend(optimum_solution[np.argmin(opt_s3)])
    pass
    # print("最优种群表",best_solution)
    # return best_solution

# 采用轮盘赌算法进行选择过程


def selection():
    # fitness_sum = sum(fitness)
    for i in range(POP_SIZE):
        population_proportion.append(fitness[i] / sum(fitness))
    pie_fitness = []
    cumsum = 0.0
    for i in range(POP_SIZE):
        pie_fitness.append(cumsum + population_proportion[i])
        cumsum += population_proportion[i]
    # 生成随机数在轮盘上选点[0, 1)
    random_selection = []
    for i in range(POP_SIZE):
        random_selection.append(random.random())
    # 选择新种群
    new_population = []
    random_selection_id = 0
    global population
    for i in range(POP_SIZE):
        while random_selection_id < POP_SIZE and random_selection[random_selection_id] < pie_fitness[i]:
            new_population.append(population[i])
            random_selection_id += 1
    population = new_population
    # print(population)          #输出新种群

# 进行交配


def crossover():
    for i in range(0, POP_SIZE-1, 2):
        if random.random() < PC:
            # 随机选择交叉点
            change_point = random.randint(1, 3)
            temp1 = []
            temp2 = []
            temp1.extend(population[i][0:change_point])
            temp1.extend(population[i+1][change_point:])
            temp2.extend(population[i+1][0:change_point])
            temp2.extend(population[i][change_point:])
            population[i] = temp1
            population[i+1] = temp2
    # print(population[i])
    # print(population[i+1])


def mutation():
    for i in range(POP_SIZE):
        if random.random() < PM:
            mutation_point = np.random.randint(1, 3)  # 随机变异点
            if mutation_point == 1:  # 如果α变异
                population[i][mutation_point] = np.random.uniform(0, 3)
            else:
                if mutation_point == 2:  # 如果β变异
                    population[i][mutation_point] = random.uniform(2, 5)
                else:  # 如果ρ变异
                    population[i, mutation_point] = round(
                        random.uniform(0, 0.5), 1)

# 以下是做图部分
# 做出平均路径长度和最优路径长度


def pictrue():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    axes[0].plot(lenAver, 'k', marker='*')
    axes[0].set_title('Average Length')
    axes[0].set_xlabel(u'iteration')
    # 线条颜色black https://blog.csdn.net/ywjun0919/article/details/8692018
    axes[1].plot(lenBest, 'k', marker='<')
    axes[1].set_title('Best Length')
    axes[1].set_xlabel(u'iteration')
    fig.savefig('Average_Best.png', dpi=500, bbox_inches='tight')
    plt.close()
    # fig.show()
    # 作出找到的最优路径图
    bestpath = best_solution[4]
    plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker='>')
    plt.xlim([0, 100])
    # x范围
    plt.ylim([0, 100])
    # y范围
    for i in range(numcity - 1):
        # 按坐标绘出最佳两两城市间路径
        m, n = int(bestpath[i]), int(bestpath[i + 1])
        print("best_path:", m, n)
        plt.plot([coordinates[m][0], coordinates[n][0]], [
                 coordinates[m][1], coordinates[n][1]], 'k')

    plt.plot([coordinates[int(bestpath[0])][0], coordinates[int(bestpath[28])][0]],
             [coordinates[int(bestpath[0])][1], coordinates[int(bestpath[27])][1]], 'b')
    ax = plt.gca()
    ax.set_title("Best Path")
    ax.set_xlabel('X_axis')
    ax.set_ylabel('Y_axis')
    plt.savefig('Best Path.png', dpi=500, bbox_inches='tight')
    plt.show()


def iters():
    init_population()
    for s in range(GENERATIONS):
        global shortest, fitness
        aco()
        calculate_fitness()
        shortest = np.array([])
        best_value()
        selection()
        fitness.clear()
        crossover()
        mutation()
        popAllList.clear()
    best_ga()


for itermax in range(19, 21):
    for zimmer in range(3):
        GENERATIONS = 15  # 遗传迭代次数
        PC = 0.88  # 交配概率
        PM = 0.3  # 变异概率
        Q = 1  # 完成率
        POP_SIZE = 10  # 种群数量6
        # itermax = 3 # 迭代总数5
        NUMANT = 10  # 蚂蚁个数5

        iter = 0  # 迭代初始
        fitness = []  # 适应度
        population = np.array([])  # 种群对应的十进制数值
        fitness_sum = np.array([])
        optimum_solution = []  # 每次迭代所获得的最优解
        best_solution = []  # 最终结果
        population_proportion = [] # 每个染色体适应度总和的比
        shortest = np.array([])
        shortestlength = np.array([])
        popAllList = []

        start = time.time()
        iters()
        end = time.time()

        print('best path:', best_solution[3],
              'time spent:', round(end-start, 1))
        # pictrue()

        if(round(end-start, 1) >= 1.1):
            data = [GENERATIONS, PC, PM, Q, POP_SIZE, itermax,
                    NUMANT, best_solution[3], round(end-start, 1)]
            f = open('aco_ga_itermax.csv', 'a', newline='')
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(data)
            f.close()
