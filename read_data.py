import proto_files.compiled.Email_pb2 as EmailPb2
from data_convert import pack_data


def readobj(path, limit):
    with open(path, 'rb') as f:
        bs = f.read(4)
        while bs:
            if limit != 0 and ct >= limit:
                break
            size = int.from_bytes(bs, 'little', signed=False)
            em = EmailPb2.Email()
            em.ParseFromString(f.read(size))
            yield em
            bs = f.read(4)


def construct_no_dup(src, dst, limit):
    ct = 0
    with open(dst, 'wb') as f:
        seen = set()
        for em in readobj(src, limit):
            # print(em)
            triple = int(em.messageid), int(getattr(em, 'from')), int(em.to)
            if triple not in seen:
                print(triple)
                seen.add(triple)
                f.write(pack_data(em))
                ct += 1
    return ct


if __name__ == '__main__':
    print(construct_no_dup('data/enron_emails.pb', 'data/enron_emails_uniq.pb', 0))
