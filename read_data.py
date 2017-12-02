path = 'data/enron_emails.pb'

import proto_files.compiled.Email_pb2 as EmailPb2


def readobj(path):
    ret = []
    with open(path, 'rb') as f:
        bs = f.read(4)
        while bs:
            size = int.from_bytes(bs, 'little', signed=False)
            em = EmailPb2.Email()
            em.ParseFromString(f.read(size))
            ret.append(em)
            bs = f.read(4)
    return ret
