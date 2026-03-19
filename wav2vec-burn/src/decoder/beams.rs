use std::collections::{BTreeSet, HashMap, hash_map};
use std::mem::{replace, take};

use crate::util::LogSpaceF32;

use super::Beam;

#[derive(Clone, Default, Debug)]
pub struct Beams {
    beams: BTreeSet<Beam>,
    prefixes: HashMap<String, Beam>,
}

impl Beams {
    pub fn add(&mut self, beam: Beam) {
        self.beams.insert(beam.clone());
        match self.prefixes.entry(beam.prefix().to_string()) {
            hash_map::Entry::Occupied(mut entry) => {
                let old_beam = replace(entry.get_mut(), beam);
                self.beams.remove(&old_beam);
            }
            hash_map::Entry::Vacant(entry) => {
                entry.insert(beam);
            }
        }
    }

    pub fn add_to_probability_with_blank(&mut self, prefix: &str, rhs: LogSpaceF32) {
        self.update(prefix, |beam| beam.add_to_probability_with_blank(rhs));
    }

    pub fn add_to_probability_without_blank(&mut self, prefix: &str, rhs: LogSpaceF32) {
        self.update(prefix, |beam| beam.add_to_probability_without_blank(rhs));
    }

    pub fn clear(&mut self) {
        self.beams.clear();
        self.prefixes.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &Beam> {
        self.beams.iter().rev()
    }

    pub fn max(&self) -> Option<&Beam> {
        self.beams.last()
    }

    pub fn extend_pruned(&mut self, other_beams: &mut Beams, beam_width: usize) {
        self.beams = other_beams.take_into_pruned(beam_width);
        self.prefixes.clear();
        let beams = self.beams.iter().cloned();
        self.prefixes.extend(beams.map(|beam| (beam.prefix().to_string(), beam)));
    }

    #[expect(clippy::mutable_key_type, reason = "Beam only uses mutability for caching")]
    fn take_into_pruned(&mut self, beam_width: usize) -> BTreeSet<Beam> {
        let pruned = match self.beams.iter().nth_back(beam_width - 1).cloned() {
            Some(last_to_keep) => self.beams.split_off(&last_to_keep),
            None => take(&mut self.beams),
        };
        self.clear();
        pruned
    }

    fn update(&mut self, prefix: &str, updater: impl FnOnce(&mut Beam)) {
        let mut entry = match self.prefixes.entry(prefix.to_string()) {
            hash_map::Entry::Occupied(entry) => {
                self.beams.remove(entry.get());
                entry
            }
            hash_map::Entry::Vacant(entry) => {
                let prefix = entry.key().clone();
                entry.insert_entry(Beam::new(prefix))
            }
        };

        updater(entry.get_mut());

        self.beams.insert(entry.get().clone());

        if entry.get().prefix() != entry.key() {
            let beam = entry.remove();
            self.prefixes.insert(beam.prefix().to_string(), beam);
        }
    }
}
