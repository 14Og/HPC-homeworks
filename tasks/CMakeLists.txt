foreach(src
    BugReduction.c
    BugParFor.c
)
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C m)
endforeach()

foreach(src
    Pi.cc
    Car.cc
)
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
endforeach()

foreach(src
    LinearSolver.cc
    LeastSquares.cc
)
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
endforeach()

add_compile_definitions(MEDIA_DIR="${PROJECT_SOURCE_DIR}/media")