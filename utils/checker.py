import torch
import logging
def CHECK255(A): # check if image in range 0-255 
    # weak checker; not gauranteer; use with caution 
    assert(A.max() >= 50 or A.max() == A.min() ), 'expect A to be in range 0 - 255, get min-{}, max={}'.format(
        A.min(), A.max())
    CHECK3D(A)
    assert(A.shape[1] == A.shape[0]), 'expect to have HW3 or HW1 input, get {}'.format(A.shape)
    assert(A.shape[-1] == 1 or A.shape[-1] == 3), 'expect to have HW3 or HW1 input, get {}'.format(A.shape)
 
def CHECKHWD(A):
    CHECK3D(A)
    assert(A.shape[1] == A.shape[0]), 'expect to have HW3 or HW1 input, get {}'.format(A.shape)
    assert(A.shape[-1] == 1 or A.shape[-1] == 3), 'expect to have HW3 or HW1 input, get {}'.format(A.shape)
 
def CHECKSAMESHAPE(A,B):
    assert(len(A.shape) == len(B.shape) and A.shape == B.shape), 'expect same shape, get {} and {}'.format(
        A.shape, B.shape)
        
def CHECKDIM(tensor, dim, val):
    if type(tensor) == list: 
        for t in tensor: 
            CHECKDIM(t, dim, val) 
    else:
        assert(len(tensor.shape) >= dim), 'expect {} to have {} dim shape {}'.format(tensor.shape, dim, val)
        assert(tensor.shape[dim] == val), 'expect {} to have {} dim shape {}'.format(tensor.shape, dim, val)
def CHECK1D(tensor):
    assert(len(tensor.shape) == 1), 'get {} {}'.format(tensor.shape, len(tensor.shape))  
    return tensor.shape 

def CHECK2D(tensor):
    assert(len(tensor.shape) == 2), 'get {} {}'.format(tensor.shape, len(tensor.shape))  
    return tensor.shape 

def CHECK4D(tensor):
    assert(len(tensor.shape) == 4), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    return tensor.shape 

def CHECK5D(tensor):
    assert(len(tensor.shape) == 5), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    return tensor.shape 

def CHECK6D(tensor):
    assert(len(tensor.shape) == 6), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    return tensor.shape 

def CHECK7D(tensor):
    assert(len(tensor.shape) == 7), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    return tensor.shape 

def CHECK3D(tensor):
    assert(len(tensor.shape) == 3), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    return tensor.shape 

def CHECKSIZE(tensor, size):
    error_msg='expected: {} ; get {}'.format(size, tensor.shape)
    if isinstance(size, torch.Tensor):
        error_msg='expected: {} ; get {}'.format(size.shape, tensor.shape)
        assert(tensor.shape == size.shape), error_msg 
    else: # list of int 
        if type(size) is not list:
            size = list(size)
        if -1 in size: # has the wildcard 
            assert(len(tensor.shape) == len(size))
            for si, s in enumerate(size):
                if s == -1: continue # skip 
                assert(tensor.shape[si] == s), error_msg
        else:
            assert(tensor.shape == torch.Size(size)), error_msg 

def CHECKTYPE(a, b, *args):
    CHECKEQ(a.type(), b.type(), '1st & 2nd')
    for ind, c in enumerate(args):
        CHECKEQ(c.type(), a.type(), 'ith %d'%(ind+3))
def CHECKEQ(a, b, s=None):
    if s is None: 
        s = ''
    if type(a) is list: 
        CHECKEQ(len(a), len(b), s)
        for i in range(len(a)): 
            CHECKEQ(a[i], b[i], s)
    elif type(a) is dict: 
        CHECKEQ(list(a.keys()), list(b.keys()), s)
        for k in a.keys(): 
            CHECKEQ(a[k], b[k], s + ' keys = {}'.format(k))

    elif torch.is_tensor(a) and torch.is_tensor(b) and a.numel() > 1: 
        assert(torch.equal(a, b)), 'a {} not == b {}; {} '.format(a.shape, b.shape, s)
        return
    else:
        assert(a == b), 'get {} {}; {}'.format(a, b, s)

def CHECKABSST(a, b):
    """ check if abs(a) < b """
    # assert(abs(a) < b), 'get {} {}'.format(a, b)
    if not (abs(a) < b):
        print('get {} {}'.format(a, b))

def CHECKEQT(a,b):
    CHECKEQ(a.shape, b.shape)
    CHECKEQ((a-b).sum(), 0)

def CHECKINFO(tensor,mode,info):
    for i in range(len(mode)):
        k = mode[i]
        if k.isdigit():
            v = int(k)
        else:
            v = info[k]
        assert(tensor.shape[i] == v), 'i{} k{} v{} get {}'.format(
                i,k,v,tensor.shape
                )
def CHECKBOX(a, b):
    a = a.cpu()
    b = b.cpu()
    assert(a.size == b.size), '{} {}'.format(a, b)
    CHECKEQ(a.bbox.shape, b.bbox.shape)
    CHECKEQ((a.bbox - b.bbox).sum(), 0)

def CHECKDEBUG(dmm_io, check_io, depth=0):
    """ check if every thing in A-list equal to B-list """
    if depth == 0 and type(dmm_io) == dict:
        return CHECKDEBUG([dmm_io], [check_io], depth=0)

    assert(type(dmm_io) == list)
    assert(type(check_io) == list)
    CHECKEQ(len(dmm_io), len(check_io))
    # logging.info(' '*(depth+1) + '>'*depth)
    for cid, (icur, iprev) in enumerate(zip(dmm_io, check_io)):
        depthstr = '   '*(depth+1) + '| ' + '[%d-%d]'%(depth, cid)
        logging.info(depthstr + '-')
        # logging.info(depthstr + 'cid: %d'%cid)
        if isinstance(icur, torch.Tensor):
            # logging.info(depthstr + 'check tensor: {}'.format(icur.shape))
            CHECKEQ(icur.shape, iprev.shape)
            CHECKEQ(icur.sum(), iprev.sum())
            # CHECKEQ(icur.mean(), iprev.mean())
            #CHECKEQ(icur.min(), iprev.min())
            #CHECKEQ(icur.max(), iprev.max())
        elif type(icur) == tuple:
            CHECKDEBUG(list(icur), list(iprev), depth+1)
        elif type(icur) == list:
            CHECKDEBUG(icur, iprev, depth+1) # list of list .. 
        elif type(icur) == dict:
            for name in icur.keys():
                logging.info(depthstr + 'dname: {}'.format(name))
                CHECKDEBUG([icur[name]], [iprev[name]], depth+1)
            #elif type(icur) == str:
            #    CHECKEQ(icur, iprev)
        else:
            logging.info(depthstr + '{}'.format(type(icur)))
            CHECKEQ(icur, iprev) # none special type
    # logging.info(' '*(depth+1) + '<'*depth)
    # logging.info(' ')

def CHECKNAN(tensor, name): 
    if torch.isnan(tensor.max()): 
        logging.error('NaN found in tensor: %s'%name)
    if torch.isinf(tensor.min()):
        logging.error('Inf found in tensor: %s'%name)
