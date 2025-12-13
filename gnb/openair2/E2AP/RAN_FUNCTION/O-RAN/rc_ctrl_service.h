#ifndef OPENAIRINTERFACE_RC_CTRL_SERVICE
#define OPENAIRINTERFACE_RC_CTRL_SERVICE

#include "openair2/E2AP/flexric/src/sm/rc_sm/ie/ir/lst_ran_param.h"
#include "openair2/E2AP/flexric/src/sm/rc_sm/ie/ir/ran_param_list.h"
#include "openair2/E2AP/flexric/src/sm/rc_sm/ie/ir/ran_parameter_value.h"
#include "openair2/E2AP/flexric/src/sm/rc_sm/ie/ir/ran_parameter_value_type.h"
#include "openair2/E2AP/flexric/src/sm/slice_sm/ie/slice_data_ie.h"

bool schedule_handover(int mod_id, size_t slices_len, ran_param_list_t* lst);

typedef enum{
  DRX_parameter_configuration_7_6_3_1 = 1,
  SR_periodicity_configuration_7_6_3_1 = 2,
  SPS_parameters_configuration_7_6_3_1 = 3,
  Configured_grant_control_7_6_3_1 = 4,
  CQI_table_configuration_7_6_3_1 = 5,
  Slice_level_PRB_quotal_7_6_3_1 = 6,
  Interference_PRB_quotal_7_6_3_1 = 7,
  handover_PRB_quotal_7_6_3_1 = 8,
} rc_ctrl_e;

/*
typedef enum {
    RRM_Policy_Ratio_List_8_4_3_7 = 1,
    RRM_Policy_Ratio_Group_8_4_3_7 = 2,
    First_Du_Map_8_4_3_7 = 3,
    Second_Du_Map_8_4_3_7 = 4,
    Third_Du_Map_8_4_3_7 = 5,
} interference_PRB_quota_param_id_e;
*/

typedef enum {
    RRM_Policy_Ratio_List_8_4_3_7 = 1,
    RRM_Policy_Ratio_Group_8_4_3_7 = 2,
    Du_Map_8_4_3_7 = 3,
    //Int_Map_8_4_3_7 = 4,
} interference_PRB_quota_param_id_e;

// typedef struct {
//   enum npf_type {pf_itself, pf_map} type;
//   union {
//     struct { float Mbps_reserved; float Mbps_reference; };
//     struct { float pct_reserved; };
//   };
// } nvs_nr_slice_param_t;

typedef enum {
    RRM_Policy_Ratio_List_8_4_3_8 = 1,
    RRM_Policy_Ratio_Group_8_4_3_8 = 2,
    Cu_Handover_8_4_3_8 = 3,
    //Int_Map_8_4_3_7 = 4,
} handover_PRB_quota_param_id_e;

#endif // OPENAIRINTERFACE_RC_CTRL_SERVICE_STYLE_2_H
