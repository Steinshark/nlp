def sort(l:list):
    sorted_list     = [] 
    for _ in range(len(l)):
        next_highest    = max(l)
        l.remove(next_highest)
        sorted_list.append(next_highest)
    return sorted_list

from torch.utils.data import DataLoader 
dl = DataLoader(dataset,batch_size=64,shuffle=)
def sort(l:list):
    nextlist    = [] 
    for i in range(len(l)):
        next_max    = float('-inf')
        i_max       = 0
        for j in range(len(l)-i):
            if l[i+j] > next_max:
                next_max = l[i+j]
                i_max   = i+j
        prev    = l[i_max]
        l[i_max]    =l[i]
        l[i]    = prev 

    return l




if __name__ == '__main__':
    j = [10,65,3,4,1,2]
    print(sort(j))