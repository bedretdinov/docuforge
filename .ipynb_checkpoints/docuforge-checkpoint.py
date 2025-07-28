import os
import uuid
import json
import pickle
from pathlib import Path
from bson import decode_all, encode
from BTrees.OOBTree import OOBTree



def read_bson_documents(f):
    data = f.read()
    return decode_all(data)


import re

class FieldExpr:
    def __init__(self, field=None):
        self.field = field
        self.op = None
        self.value = None
        self.extra = None  # Для дополнительных аргументов (например, between)
        self.left = None
        self.right = None
        self.bool_op = None  # 'and', 'or'

    def _make(self, op, value, extra=None):
        out = FieldExpr(self.field)
        out.op = op
        out.value = value
        out.extra = extra
        return out

    # --- Базовые операторы ---
    def __eq__(self, value): return self._make("==", value)
    def __ne__(self, value): return self._make("!=", value)
    def __lt__(self, value): return self._make("<", value)
    def __le__(self, value): return self._make("<=", value)
    def __gt__(self, value): return self._make(">", value)
    def __ge__(self, value): return self._make(">=", value)

    # --- Логические ---
    def __and__(self, other):
        expr = FieldExpr()
        expr.bool_op = 'and'
        expr.left = self
        expr.right = other
        return expr

    def __or__(self, other):
        expr = FieldExpr()
        expr.bool_op = 'or'
        expr.left = self
        expr.right = other
        return expr

    # --- Дополнительные фильтры ---
    def isin(self, values): return self._make("in", values)
    def notin(self, values): return self._make("not in", values)

    def regex(self, pattern): return self._make("regex", pattern)
    def not_regex(self, pattern): return self._make("not_regex", pattern)

    def is_none(self): return self._make("is_none", None)
    def not_none(self): return self._make("not_none", None)

    def startswith(self, prefix): return self._make("startswith", prefix)
    def endswith(self, suffix): return self._make("endswith", suffix)
    def contains(self, substring): return self._make("contains", substring)
    def not_contains(self, substring): return self._make("not_contains", substring)

    def between(self, low, high): return self._make("between", low, high)
    def not_between(self, low, high): return self._make("not_between", low, high)

    def is_type(self, t): return self._make("is_type", t)

    def len_eq(self, length): return self._make("len_eq", length)
    def len_gt(self, length): return self._make("len_gt", length)
    def len_lt(self, length): return self._make("len_lt", length)

    # --- Тестирование ---
    def test(self, doc):
        if self.bool_op:
            if self.bool_op == 'and':
                return self.left.test(doc) and self.right.test(doc)
            if self.bool_op == 'or':
                return self.left.test(doc) or self.right.test(doc)

        value = doc.get(self.field)

        if self.op == "==": return value == self.value
        if self.op == "!=": return value != self.value
        if self.op == "<":  return value < self.value
        if self.op == "<=": return value <= self.value
        if self.op == ">":  return value > self.value
        if self.op == ">=": return value >= self.value

        if self.op == "in": return value in self.value
        if self.op == "not in": return value not in self.value

        if self.op == "regex": return bool(re.search(self.value, str(value)))
        if self.op == "not_regex": return not bool(re.search(self.value, str(value)))

        if self.op == "is_none": return value is None
        if self.op == "not_none": return value is not None

        if self.op == "startswith": return isinstance(value, str) and value.startswith(self.value)
        if self.op == "endswith": return isinstance(value, str) and value.endswith(self.value)
        if self.op == "contains": return isinstance(value, str) and self.value in value
        if self.op == "not_contains": return isinstance(value, str) and self.value not in value

        if self.op == "between": return self.value <= value <= self.extra
        if self.op == "not_between": return not (self.value <= value <= self.extra)

        if self.op == "is_type": return isinstance(value, self.value)

        if self.op == "len_eq": return len(value) == self.value
        if self.op == "len_gt": return len(value) > self.value
        if self.op == "len_lt": return len(value) < self.value

        return False


class DocuForge:
    def __init__(self, db_path='dbdata'):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

    def __getattr__(self, name):
        return Collection(self, name)

    def col(self, name):
        return FieldExpr(name)
  
    def _collection_path(self, collection):
        p = self.db_path / collection
        p.mkdir(exist_ok=True)
        return p

    def _wal_path(self, collection):
        return self._collection_path(collection) / "wal.bson"

    def _btree_path(self, collection, field):
        return self._collection_path(collection) / f"index_{field}.btree"

    def _next_data_file(self, collection):
        coll_path = self._collection_path(collection)
        files = sorted(
            f for f in coll_path.glob("*.bson") if f.name != "wal.bson"
        )
    
        for f in reversed(files):
            with open(f, "rb") as fh:
                count = sum(1 for _ in read_bson_documents(fh) if not _get(_, "_deleted"))
                if count < 1000:
                    return f
    
        return coll_path / f"data_{uuid.uuid4().hex[:8]}.bson"

    def _write_to_file(self, path, doc):
        with open(path, "ab") as f:
            f.write(encode(doc))

    def _flush_wal(self, collection):
        wal_path = self._wal_path(collection)
        if not wal_path.exists():
            return
        with open(wal_path, "rb") as f:
            docs = read_bson_documents(f) 
        for doc in docs:
            if doc.get("_op") == "insert":
                data_path = self._next_data_file(collection)
                print('_next_data_file',data_path)
                self._write_to_file(data_path, doc)
                self._update_indexes(collection, doc)
            elif doc.get("_op") == "update":
                self._apply_update(collection, doc)
            elif doc.get("_op") == "delete":
                self._apply_delete(collection, doc)
        wal_path.unlink()

    def _update_indexes(self, collection, doc):
        for key, value in doc.items():
            if key.startswith("_"): continue
            path = self._btree_path(collection, key)
            if path.exists():
                with open(path, "rb") as f:
                    index = pickle.load(f)
            else:
                index = OOBTree()
            index.setdefault(value, []).append(doc["_id"])
            with open(path, "wb") as f:
                pickle.dump(index, f)

    def insert_one(self, collection, doc):
        doc["_id"] = str(uuid.uuid4())
        doc["_op"] = "insert"
        self._write_to_file(self._wal_path(collection), doc)
        return doc["_id"]

    def _apply_update(self, collection, patch):
        all_docs = list(self.find(collection, {"_id": patch["_id"]}))
        if not all_docs:
            return
        orig = all_docs[0]
        orig.update(patch["_set"])
        orig["_op"] = "insert"
        self._write_to_file(self._next_data_file(collection), orig)
        self._update_indexes(collection, orig)

    def _apply_delete(self, collection, patch):
        patch_doc = {"_id": patch["_id"], "_deleted": True, "_op": "insert"}
        self._write_to_file(self._next_data_file(collection), patch_doc)

    def update_one(self, collection, _id, updates):
        patch = {"_op": "update", "_id": _id, "_set": updates}
        self._write_to_file(self._wal_path(collection), patch)

    def delete_one(self, collection, _id):
        patch = {"_op": "delete", "_id": _id}
        self._write_to_file(self._wal_path(collection), patch)

    def find(self, collection, filter_expr=None):
        self._flush_wal(collection)
        coll_path = self._collection_path(collection)
        for bson_file in sorted(coll_path.glob("*.bson")):
            with open(bson_file, "rb") as f:
                for doc in read_bson_documents(f):
                    if doc.get("_deleted"):
                        continue
                    if filter_expr and not filter_expr.test(doc):
                        continue
                    yield doc
 

    def defragment_collection(self, collection):
        self._flush_wal(collection)
        all_docs = [doc for doc in self.find(collection)] 
        for file in self._collection_path(collection).glob("*.bson"):
            file.unlink()
        for file in self._collection_path(collection).glob("*.btree"):
            file.unlink()
        for doc in all_docs:
            self.insert_one(collection, doc)


class Collection:
    def __init__(self, db, name):
        self.db = db
        self.__name = name


    def __getattr__(self, name):
        return FieldExpr(name)
    
    def insert_one(self, doc):
        return self.db.insert_one(self.__name, doc)

    def find(self, filter_expr=None):
        return self.db.find(self.__name, filter_expr)

    def update_one(self, _id, updates):
        return self.db.update_one(self.__name, _id, updates)

    def delete_one(self, _id):
        return self.db.delete_one(self.__name, _id)


def _get(d, k):
    return d[k] if k in d else None
