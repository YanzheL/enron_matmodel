path = 'data/enron_emails.pb'

import proto_files.compiled.Email_pb2 as EmailPb2
from data_convert import pack_data


def readobj(path):
    seen = set()
    ret = []
    ct = 0
    with open(path, 'rb') as f:
        bs = f.read(4)
        while bs:
            size = int.from_bytes(bs, 'little', signed=False)
            em = EmailPb2.Email()
            em.ParseFromString(f.read(size))
            triple = em.messageid, getattr(em, 'from'), em.to
            if triple not in seen:
                seen.add(triple)
                ret.append(em)
                ct += 1
            bs = f.read(4)
    with open('data/enron_emails_uniq.pb', 'wb'):
        for e in ret:
            f.write(pack_data(e))
    print(ct)
    return ret


if __name__ == '__main__':
    readobj(path)
