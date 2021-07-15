use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct Memdb {
    hashmap: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl Memdb {
    /// Create a new instance.
    pub fn open() -> Self {
        Self {
            hashmap: Arc::new(RwLock::new(HashMap::<String, Vec<u8>>::new())),
        }
    }

    /// Set a value in the database.
    pub fn set(&self, key: String, value: impl AsRef<[u8]>) -> Option<Vec<u8>> {
        let hashmap = self.hashmap.clone();
        let mut hashmap = hashmap.write();
        hashmap.insert(key, value.as_ref().to_owned())
    }

    /// Get a value from the database.
    #[must_use]
    pub fn get(&self, key: String) -> Option<Vec<u8>> {
        let hashmap = &self.hashmap.read();
        hashmap.get(&key).cloned()
    }

    /// Delete a value from the database.
    pub fn del(&self, key: String) -> Option<Vec<u8>> {
        let hashmap = &mut self.hashmap.write();
        hashmap.remove(&key)
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_memdb() {
        let memdb = Memdb::open();

        memdb.set("key1".into(), "123".as_bytes());
        assert_eq!(memdb.get("key1".into()).unwrap(), "123".as_bytes().to_vec());
        memdb.del("key1".into());
        assert_eq!(memdb.get("key1".into()), None);
    }
}
