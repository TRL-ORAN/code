#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "openair2/LAYER2/NR_MAC_gNB/slicing/nr_slicing.h"
#include "openair2/RRC/NR/rrc_gNB_UE_context.h"
#include "rc_ctrl_service.h"

bool schedule_handover(int mod_id, size_t ue_len, ran_param_list_t* lst)
{
  gNB_MAC_INST *nrmac = RC.nrmac[mod_id];
  assert(nrmac);

  //pthread_mutex_lock(&nrmac->UE_info.mutex);
  int num_map = 5;
  NR_UEs_t *UE_info = &RC.nrmac[mod_id]->UE_info;
  double du_prb_ratio[ue_len][num_map];
  for (size_t i = 0; i < ue_len; ++i) {

    //NR_UE_info_t *UE = UE_info->list[i];

    lst_ran_param_t* RRM_Policy_Ratio_Group = &lst->lst_ran_param[i];
    //Bug in rc_enc_asn.c:1003, asn didn't define ran_param_id for lst_ran_param_t...
    //assert(RRM_Policy_Ratio_Group->ran_param_id == RRM_Policy_Ratio_Group_8_4_3_6 && "wrong RRM_Policy_Ratio_Group id");
    assert(RRM_Policy_Ratio_Group->ran_param_struct.sz_ran_param_struct ==  10 && "wrong RRM_Policy_Ratio_Group->ran_param_struct.sz_ran_param_struct");
    assert(RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct != NULL && "NULL RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct");
    ///// interferace map /////
    for (int j = 0; j <num_map; j++){
      seq_ran_param_t* Du_Map = &RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct[j];
      assert(Du_Map->ran_param_id == Du_Map_8_4_3_7 && "wrong First_Du_Map_8_4_3_7 id");
      assert(Du_Map->ran_param_val.type == ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE && "wrong First_Du_Map type");
      assert(Du_Map->ran_param_val.flag_false != NULL && "NULL Min_PRB_Policy_Ratio->ran_param_val.flag_false");
      assert(Du_Map->ran_param_val.flag_false->type == INTEGER_RAN_PARAMETER_VALUE && "wrong First_Du_Map->ran_param_val.flag_false type");
      //int64_t first_prb_ratio = Du_Map->ran_param_val.flag_false->int_ran;
      
      du_prb_ratio[i][j] = Du_Map->ran_param_val.flag_false->real_ran;
      //printf("%f\n", du_prb_ratio[i][j]);
      //UE.interference_map[j] = du_prb_ratio[i][j];

      //LOG_E(NR_MAC, "du_map %f\n", du_prb_ratio[i][j]);
    }
  }
  
  int ue_id = 0;

  UE_iterator(UE_info->list, UE) {
    for (int j = 0; j <num_map; j++){
      //LOG_E(NR_MAC, "du_map %f\n", du_prb_ratio[ue_id][j]);
      UE->interference_map[j] = du_prb_ratio[ue_id][j];
      //printf("UE->interference_map[j] %f\n", UE->interference_map[j]);
      LOG_E(NR_MAC, "UE->interference_map[j] %f\n", UE->interference_map[j]);
    }
    ue_id ++;
  }

  
    ///// send map to mac layer /////
  return true;
}