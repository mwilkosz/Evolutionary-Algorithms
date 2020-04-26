import matplotlib.pyplot as plt
import numpy as np 

token = open('model3.txt','r')
linestoken=token.readlines()
input_column_number = 0
output_column_number = 1
input_data=[]
output_data=[]
for x in linestoken:
    input_data.append(x.split()[input_column_number])
    output_data.append(x.split()[output_column_number])
token.close()

input_data=[float(i) for i in input_data]
output_data=[float(i) for i in output_data]

x_values = np.random.uniform(-10,10,size = (100, 3))
sigma = np.random.uniform(0,10,size = (100,3))

#Constant values
u_size = 100
lambda_size = 5
generations = 100

tao1 = 1/(np.sqrt(2*6))
tao2 = 1/np.sqrt(2*(np.sqrt(6)))

#Main loop
for i in range(generations):

    lambda_x_values = np.empty([u_size*lambda_size,3])
    lambda_sigma_values = np.empty([u_size*lambda_size,3])
        
    iteration = 0
    for i in range(u_size*lambda_size):

        random_chromosome=np.random.randint(0,u_size)
        #Update x_values
        r = np.zeros(3,)
        r1 = np.array(np.random.normal(0,sigma[random_chromosome,0]))
        r2 = np.array(np.random.normal(0,sigma[random_chromosome,1]))
        r3 = np.array(np.random.normal(0,sigma[random_chromosome,2]))
        lambda_x = np.array([(x_values[random_chromosome,0]+r1),(x_values[random_chromosome,1]+r2),(x_values[random_chromosome,2]+r3)])
        lambda_x_values[iteration] = lambda_x
        
        #Update sigma
        r_sigm1 = tao1 * np.random.normal()
        r_sigm2 = tao2 * np.random.normal(size=3)
        sigma_a1 = sigma[random_chromosome,0] * np.exp(r_sigm1) * np.exp(r[0])
        sigma_b1 = sigma[random_chromosome,1] * np.exp(r_sigm1) * np.exp(r[1])
        sigma_c1 = sigma[random_chromosome,2] * np.exp(r_sigm1) * np.exp(r[2])
        lambda_sigma = np.zeros(3,)
        lambda_sigma = np.array([sigma_a1,sigma_b1,sigma_c1])
        lambda_sigma_values[iteration] = lambda_sigma
        iteration+=1

    #Calculating MSE of population
    mse_array = np.zeros([u_size*lambda_size,2])         
    index = 0                                               
    for i in lambda_x_values:
        output = []
        a = i[0] 
        b = i[1] 
        c = i[2] 
        for j in input_data:  
            output.append(a*((j**2)-(b*np.cos(c*np.pi*j))))
        total = 0
        t = 0
        for k in output:
            total += ((output_data[t]-output[t])**2)
            t += 1
        mse = 1/len(output)*total

        mse_array[index] = (index,mse)
        index += 1

    #Choosing best chromosomes to create parents populations 
    ind = np.argsort(mse_array[:,-1]) 
    b = mse_array[ind]
    mse_array = b[0:u_size,:] 
    # best_mse = np.min(mse_array[:,1])
    best_mse=mse_array[np.argmin(mse_array[:, 0]), :]    

    #Deleting worse chromosomes
    to_delete = []
    index = 0
    for i in lambda_x_values:    
        if index in mse_array[:,0]:
            pass
        else:
            to_delete.append(index)
        index += 1
    lambda_x_values = np.delete(lambda_x_values,to_delete,axis = 0)        
    lambda_sigma_values = np.delete(lambda_sigma_values,to_delete,axis = 0)

    #Updating parents population
    x_values = np.zeros([u_size,3])
    sigma = np.zeros([u_size,3])
    x_values = lambda_x_values
    sigma = lambda_sigma_values

print("Obtained MSE error is {}".format(best_mse[1]))
abc=x_values[int(best_mse[0])]
output=[]
a=abc[0]
b=abc[1]
c=abc[2]

for i in input_data:  
    output.append(a*((i**2)-(b*np.cos(c*np.pi*i))))

#Plot drawing
plt.plot(input_data,output,'b',label="Obtained data")
plt.plot(input_data,output_data,'r--',label="Output data")
plt.title("Evolution strategy (μ,λ)")
plt.xlabel("Coefficents A: {} B: {} C: {}\nMSE: {}\nIterations: {}".format(a,b,c,best_mse[1]),fontsize=8)
plt.legend(loc='upper left')           


plt.show()



