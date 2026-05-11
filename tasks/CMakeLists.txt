foreach(src
    PingPong.cc  
    PingPong2.cc
    Automata.cc
)
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
endforeach()

target_compile_definitions(Automata PRIVATE HIST="${PROJECT_SOURCE_DIR}/assets/automata_history.csv")
