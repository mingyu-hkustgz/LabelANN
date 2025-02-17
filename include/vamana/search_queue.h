#ifndef SEARCH_QUQUE
#define SEARCH_QUQUE

#include <vector>
#include <memory>
#include "config.h"


namespace ANNS {

    // for storing each candidate, prefer those with minimal distances
    struct Candidate {
        IdxType id;
        float distance;
        bool expanded;

        Candidate() = default;

        Candidate(IdxType a, float b) : id{a}, distance{b}, expanded(false) {}

        inline bool operator<(const Candidate &other) const {
            return distance < other.distance || (distance == other.distance && id < other.id);
        }

        inline bool operator==(const Candidate &other) const { return (id == other.id); }
    };


    // search queue for ANNS, preserve the closest vectors
    class SearchQueue {

    public:
        SearchQueue() : _size(0), _capacity(0), _cur_unexpanded(0) {};

        ~SearchQueue() = default;

        // size
        int32_t size() const { return _size; };

        int32_t capacity() const { return _capacity; };

        void reserve(int32_t capacity) {
            if (capacity + 1 > _data.size())
                _data.resize(capacity + 1);
            _capacity = capacity;
        }

        // read and write
        Candidate operator[](int32_t idx) const { return _data[idx]; }

        Candidate &operator[](int32_t idx) { return _data[idx]; };

        bool exist(IdxType id) const {
            for (int32_t i = 0; i < _size; i++)
                if (_data[i].id == id)
                    return true;
            return false;
        }

        void insert(IdxType id, float distance) {
            Candidate new_candidate(id, distance);
            if (_size == _capacity && _data[_size - 1] < new_candidate)
                return;

            // binary search
            int32_t lo = 0, hi = _size;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (new_candidate < _data[mid])
                    hi = mid;
                else if (UNLIKELY(_data[mid].id == new_candidate.id))   // ensure the same id is not in the set
                    return;
                else
                    lo = mid + 1;
            }

            // move the elements
            if (lo < _capacity)
                std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Candidate));
            _data[lo] = new_candidate;

            // update size and currently unexpanded candidate
            if (_size < _capacity)
                _size++;
            if (lo < _cur_unexpanded)
                _cur_unexpanded = lo;
        }

        void clear() {
            _size = 0;
            _cur_unexpanded = 0;
        };

        // expand
        bool has_unexpanded_node() const { return _cur_unexpanded < _size; };

        const Candidate &get_closest_unexpanded() {
            _data[_cur_unexpanded].expanded = true;
            auto pre = _cur_unexpanded;
            while (_cur_unexpanded < _size && _data[_cur_unexpanded].expanded)
                _cur_unexpanded++;
            return _data[pre];
        }

    private:

        int32_t _size, _capacity, _cur_unexpanded;
        std::vector<Candidate> _data;
    };
}

#endif // SEARCH_QUQUE