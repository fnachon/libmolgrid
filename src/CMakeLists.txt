
set( LIBMOLGRID_SOURCES
 libmolgrid.cpp
 atom_typer.cpp
 example.cpp
 exampleref_providers.cpp
 example_extractor.cpp
 example_provider.cpp
 example_dataset.cpp
 grid_maker.cpp
 grid_maker_gpu.mm
 coordinateset.cpp
 coordinateset_gpu.mm
 coord_cache.cpp
 transform.cpp
 transform_gpu.mm
 grid_io.cpp
 grid_interpolater.cpp
 grid_interpolater_gpu.mm
 cartesian_grid.cpp
 string_cache.cpp
 metal_context.mm
)

set( LIBMOLGRID_HEADERS
 ../include/libmolgrid/libmolgrid.h
 ../include/libmolgrid/grid.h
 ../include/libmolgrid/managed_grid.h
 ../include/libmolgrid/quaternion.h
 ../include/libmolgrid/transform.h
 ../include/libmolgrid/atom_typer.h
 ../include/libmolgrid/example.h
 ../include/libmolgrid/coordinateset.h
 ../include/libmolgrid/exampleref_providers.h
 ../include/libmolgrid/example_extractor.h
 ../include/libmolgrid/example_provider.h
 ../include/libmolgrid/example_dataset.h
 ../include/libmolgrid/grid_maker.h
  ../include/libmolgrid/grid_interpolater.h
 ../include/libmolgrid/coord_cache.h
 ../include/libmolgrid/common.h
 ../include/libmolgrid/grid_io.h
 ../include/libmolgrid/cartesian_grid.h
 ../include/libmolgrid/string_cache.h
 ../include/libmolgrid/metal_context.h
)

set ( LIBMOLGRID_HEADERS ${LIBMOLGRID_HEADERS}  PARENT_SCOPE)

if(BUILD_SHARED)
    add_library(libmolgrid_shared SHARED ${LIBMOLGRID_HEADERS} ${LIBMOLGRID_SOURCES})
    SET_TARGET_PROPERTIES(libmolgrid_shared PROPERTIES OUTPUT_NAME molgrid)
    target_link_libraries(libmolgrid_shared
        ${OPENBABEL3_LIBRARIES}
        ${Boost_LIBRARIES}
        ${ZLIB_LIBRARIES}
        "-framework Metal"
        "-framework Foundation"
    )
    install(TARGETS libmolgrid_shared DESTINATION lib)
endif()

if(BUILD_STATIC)
    add_library(libmolgrid_static STATIC ${LIBMOLGRID_HEADERS} ${LIBMOLGRID_SOURCES})
    SET_TARGET_PROPERTIES(libmolgrid_static PROPERTIES OUTPUT_NAME molgrid POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(libmolgrid_static
        ${OPENBABEL3_LIBRARIES}
        ${Boost_LIBRARIES}
        ${ZLIB_LIBRARIES}
        "-framework Metal"
        "-framework Foundation"
    )
    install(TARGETS libmolgrid_static DESTINATION lib)
endif()

# Compile Metal shaders → libmolgrid_shaders.metallib
find_program(XCRUN xcrun REQUIRED)

set(METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/metal/shaders.metal)
set(METAL_AIR    ${CMAKE_CURRENT_BINARY_DIR}/shaders.air)
set(METAL_LIB    ${CMAKE_CURRENT_BINARY_DIR}/libmolgrid_shaders.metallib)

# Pass the metallib path to metal_context.mm at compile time
target_compile_definitions(libmolgrid_shared PRIVATE
    METAL_LIBRARY_PATH="${CMAKE_CURRENT_BINARY_DIR}/libmolgrid_shaders.metallib")
if(BUILD_STATIC)
    target_compile_definitions(libmolgrid_static PRIVATE
        METAL_LIBRARY_PATH="${CMAKE_CURRENT_BINARY_DIR}/libmolgrid_shaders.metallib")
endif()

add_custom_command(
    OUTPUT  ${METAL_AIR}
    COMMAND ${XCRUN} -sdk macosx metal
            -std=metal3.0
            -O2
            -c ${METAL_SOURCE}
            -o ${METAL_AIR}
    DEPENDS ${METAL_SOURCE}
    COMMENT "Compiling Metal shaders to AIR"
)

add_custom_command(
    OUTPUT  ${METAL_LIB}
    COMMAND ${XCRUN} -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
    DEPENDS ${METAL_AIR}
    COMMENT "Linking Metal AIR → metallib"
)

add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIB})

if(BUILD_SHARED)
    add_dependencies(libmolgrid_shared metal_shaders)
endif()
if(BUILD_STATIC)
    add_dependencies(libmolgrid_static metal_shaders)
endif()

install(FILES ${METAL_LIB} DESTINATION lib)
install(FILES ${LIBMOLGRID_HEADERS} ${PROJECT_BINARY_DIR}/include/libmolgrid/config.h
        DESTINATION include/libmolgrid)
