foreach(src
    PingPong.cc  
    PingPong2.cc
    Automata.cc
)
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
    target_include_directories(${target} PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
endforeach()
