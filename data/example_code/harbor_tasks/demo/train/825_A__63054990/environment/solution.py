n=int(input())
s=input()
ones=0
newn=1
ans=""
i=0

while(i<n):
    if i==n-1:
        if s[i]=="1":
            ones+=1
        ans+=str(ones)
    if(s[i]=="1"):
        ones+=1
        i+=1
    else:
        ans+=str(ones)
        ones=0
        i+=1
print(ans)