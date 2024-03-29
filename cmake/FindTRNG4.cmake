# Copyright (c) 2020 Dongkyu Kim (dkkim1005@gmail.com)

IF (NOT TRNG4_FOUND)
  # Setting default paths
  IF (NOT TRNG4_INCLUDE_DIR)
    SET (TRNG4_INCLUDE_DIR /usr/local/include)
  ENDIF (NOT TRNG4_INCLUDE_DIR)
  MESSAGE (STATUS "TRNG4_INCLUDE_DIR (DEFAULT) : ${TRNG4_INCLUDE_DIR}")
  IF (NOT TRNG4_LIBRARY_DIR)
    SET (TRNG4_LIBRARY_DIR /usr/local/lib/)
  ENDIF (NOT TRNG4_LIBRARY_DIR)
  MESSAGE (STATUS "TRNG4_LIBRARY_DIR (DEFAULT) : ${TRNG4_LIBRARY_DIR}")

  # Check whether header files are in ${TRNG4_INCLUDE_DIR} or not.
  MESSAGE (STATUS "Checking for trng4 in the default path")
  SET (HEADER_FILES "yarn5.hpp;uniform01_dist.hpp;uniform_int_dist.hpp")
  FOREACH (HEADER_FILE ${HEADER_FILES})
    MESSAGE (STATUS "  Looking for ${HEADER_FILE}")
    FIND_PATH (HEADER_FILE_PATH NAMES ${HEADER_FILE} HINTS "${TRNG4_INCLUDE_DIR}/trng")
    IF (HEADER_FILE_PATH)
      MESSAGE (STATUS "  Looking for ${HEADER_FILE} - found")
    ELSE ()
      MESSAGE (NOTICE "  ${HEADER_FILE} is not detected at the path: ${TRNG4_INCLUDE_DIR}")
      SET (TRNG4_FOUND FALSE)
    ENDIF (HEADER_FILE_PATH)
  ENDFOREACH ()

  # suffix of the dynamic library 
  IF (APPLE)
    SET (DYLIBSUFFIX dylib)
  ELSEIF (UNIX)
    SET (DYLIBSUFFIX so)
  ENDIF ()

  # Check library
  FIND_LIBRARY (TRNG4_LIBRARY_DIR NAMES trng4 libtrng4 HINTS ${TRNG4_LIBRARY_DIR})
  IF (EXISTS ${TRNG4_LIBRARY_DIR}/libtrng4.${DYLIBSUFFIX})
    MESSAGE (STATUS "TRNG4 library - found")
  ELSE()
    MESSAGE (NOTICE "  Library trng4 is not detected at the path: ${TRNG4_LIBRARY_DIR}")
    SET (TRNG4_FOUND FALSE)
  ENDIF (EXISTS ${TRNG4_LIBRARY_DIR}/libtrng4.${DYLIBSUFFIX})

  IF (TRNG4_FOUND MATCHES FALSE)
    # repository stored in GitHub 
    SET (TRNG4_GIT_REPOSITORY https://github.com/rabauke/trng4.git)
    # tag id of trng4 library compatible to cuda 10
    SET (TRNG4_GIT_TAG_ID v4.22)
    IF (APPLE OR UNIX)
      # download trng4 source files from the external git repository
      IF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trng4/CMakeLists.txt)
        EXECUTE_PROCESS (COMMAND git clone ${TRNG4_GIT_REPOSITORY} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        EXECUTE_PROCESS (COMMAND git checkout tags/${TRNG4_GIT_TAG_ID} -b temp WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/trng4)
      ENDIF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trng4/CMakeLists.txt)
      # The library will be installed at "trng4.build".
      IF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trng4.build)
        EXECUTE_PROCESS (COMMAND mkdir ${CMAKE_CURRENT_SOURCE_DIR}/trng4.build WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      ENDIF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trng4.build)
      IF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trng4.build/trng/libtrng4.${DYLIBSUFFIX})
       MESSAGE (NOTICE "  trng4 library is being installed at the current source directory.")
        EXECUTE_PROCESS (COMMAND ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/trng4 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/trng4.build)
        EXECUTE_PROCESS (COMMAND make WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/trng4.build)
        IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trng4.build/trng/libtrng4.${DYLIBSUFFIX})
          MESSAGE (NOTICE "  Installation of trng4 library is now finished.")
        ELSE ()
          MESSAGE (FATAL_ERROR "  trng4 library is not installed yet.")
        ENDIF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trng4.build/trng/libtrng4.${DYLIBSUFFIX})
      ELSE ()
        MESSAGE (STATUS "TRNG4 library (${CMAKE_CURRENT_SOURCE_DIR}/trng4.build/trng/libtrng4.${DYLIBSUFFIX}) - found")
      ENDIF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trng4.build/trng/libtrng4.${DYLIBSUFFIX})
      SET (TRNG4_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/trng4/)
      SET (TRNG4_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/trng4.build/trng)
    ENDIF ()
  ENDIF (TRNG4_FOUND MATCHES FALSE)

  # final
  SET (TRNG4_FOUND TRUE)
  INCLUDE_DIRECTORIES (SYSTEM ${TRNG4_INCLUDE_DIR})
  LINK_DIRECTORIES(${TRNG4_LIBRARY_DIR})
  SET (TRNG4_LIBRARIES trng4)
ENDIF (NOT TRNG4_FOUND)
