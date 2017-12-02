path = 'data/enron_emails.pb'

import proto_files.compiled.Email_pb2 as EmailPb2


def readobj(path):
    ret = []
    ct=0
    with open(path, 'rb') as f:
        bs = f.read(4)
        while bs:
            size = int.from_bytes(bs, 'little', signed=False)
            em = EmailPb2.Email()
            em.ParseFromString(f.read(size))
            ret.append(em)
            if ct>255600:
                print(em)
            ct+=1
            bs = f.read(4)
    print(ct)
    return ret

if __name__ == '__main__':
    readobj(path)