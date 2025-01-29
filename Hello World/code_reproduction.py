L=list(range(1,10,1))
print(L,'数组和为:',sum(L))
L=['p','y','t','h','o','n']
print(L[4])
print(L[0:3])
print(L[0:4:2])
print(L[-1])
L[1]='Y'
print(L)
L[2 : 5] =['T','H','0']
print(L)
L[-1]='N'
print(L)
L.append('test1')
print(L)
L.insert(3,'test2')
print(L)
L.pop( )
print(L)
L.pop(3)
print(L)
nums = [9,6,1,4,2]
print(sorted(nums))
print(nums)
nums.sort()
print(nums)
nums.sort(reverse=True)
print(nums)
s=0
for i in range(1,11):
    s=s+1/i**2
print(s)
L=[2*x+2 for x in range(50)]
print(L)
x=eval(input('x='))
if x>0:
    a=x
else:
    a=-x
print('|x|=',a)