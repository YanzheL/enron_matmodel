import proto_files.compiled.Email_pb2 as EmailPb2
from data_convert import pack_data


def readobj(path):
    ct = 0
    with open(path, 'rb') as f:
        bs = f.read(4)
        while bs:
            size = int.from_bytes(bs, 'little', signed=False)
            em = EmailPb2.Email()
            em.ParseFromString(f.read(size))
            yield em
            ct += 1
            bs = f.read(4)
    print(ct)


if __name__ == '__main__':
    with open('data/enron_emails_uniq.pb', 'wb') as f:
        seen = set()
        for em in readobj('data/enron_emails.pb'):
            triple = em.messageid, getattr(em, 'from'), em.to
            if triple not in seen:
                seen.add(triple)
                f.write(pack_data(em))
