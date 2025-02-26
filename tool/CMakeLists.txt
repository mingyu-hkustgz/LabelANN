add_executable(generate_base_labels generate_base_labels.cpp
        ../include/IndexLabelElastic.h
)
target_link_libraries(generate_base_labels Boost::program_options)

add_executable(generate_query_labels generate_query_labels.cpp)
target_link_libraries(generate_query_labels Boost::program_options)