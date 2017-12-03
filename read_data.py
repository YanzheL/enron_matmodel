import proto_files.compiled.Email_pb2 as EmailPb2
from data_convert import pack_data
from collections import defaultdict

def readobj(path, limit):
    with open(path, 'rb') as f:
        ct = 0
        bs = f.read(4)
        while bs:
            if limit != 0 and ct >= limit:
                break
            size = int.from_bytes(bs, 'little', signed=False)
            em = EmailPb2.Email()
            em.ParseFromString(f.read(size))
            yield em
            ct += 1
            bs = f.read(4)


def construct_no_dup(src, dst, limit):
    ct = 0
    with open(dst, 'wb') as f:
        seen = set()
        for em in readobj(src, limit):
            # print(em)
            triple = int(em.messageid), int(em.sender), int(em.recipient)
            if triple not in seen:
                # print(triple)
                seen.add(triple)
                f.write(pack_data(em))
                ct += 1
    return ct


def classify_by_sender(src, dst, limit):
    ct = 0
    data = defaultdict(list)
    seen = set()
    for em in readobj(src, limit):
        # print(em)
        spec = int(em.messageid), int(em.sender)
        if spec not in seen:
            # print(triple)
            seen.add(spec)
            data[spec[1]].append(em)
            ct += 1
    for k, ems in data.items():
        with open(dst % k, 'wb') as f:
            for em in ems:
                f.write(pack_data(em))

    return ct


if __name__ == '__main__':
    print(classify_by_sender('data/enron_emails.pb', 'data/by_people_%d.pb', 0))
