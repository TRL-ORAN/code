add_executable(xapp_mac_ctrl
  mac_ctrl.c
  ../../../../src/util/alg_ds/alg/defer.c
  )

 target_link_libraries(xapp_mac_ctrl
 						PUBLIC
 						e42_xapp
 						-pthread
 						-lsctp
 						-ldl
 						)


add_executable(xapp_rc_slice_ctrl
	xapp_rc_slice_ctrl.c
	../../../../src/util/alg_ds/alg/defer.c
	)

target_link_libraries(xapp_rc_slice_ctrl
											PUBLIC
											e42_xapp
											-pthread
											-lsctp
											-ldl
											)

add_executable(xapp_rb_ctrl
	xapp_rb_ctrl.c
	../../../../src/util/alg_ds/alg/defer.c
	)

target_link_libraries(xapp_rb_ctrl
											PUBLIC
											e42_xapp
											-pthread
											-lsctp
											-ldl
											)
											
add_executable(xapp_combin_ctrl
	xapp_combin_ctrl.c
	../../../../src/util/alg_ds/alg/defer.c
	)

target_link_libraries(xapp_combin_ctrl
											PUBLIC
											e42_xapp
											-pthread
											-lsctp
											-ldl
											)


add_executable(xapp_handover
	xapp_handover.c
	../../../../src/util/alg_ds/alg/defer.c
	)
										
target_link_libraries(xapp_handover
											PUBLIC
											e42_xapp
											-pthread							
											-lsctp
											-ldl
											)


