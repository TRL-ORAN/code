
 #include "../../../../src/xApp/e42_xapp_api.h"
 #include "../../../../src/sm/rc_sm/ie/ir/ran_param_struct.h"
 #include "../../../../src/sm/rc_sm/ie/ir/ran_param_list.h"
 #include "../../../../src/util/time_now_us.h"
 #include "../../../../src/util/alg_ds/ds/lock_guard/lock_guard.h"
 #include "../../../../src/sm/rc_sm/rc_sm_id.h"
 #include <stdlib.h>
 #include <stdio.h>
 #include <time.h>
 #include <unistd.h>
 
 //#######################################rb_ctrl#################################
 int nodes_id = 0;
 int nb_id = 0;
 
 //***************************************kpm_moni********************************
 static
 uint64_t const period_ms = 10;
 static
 pthread_mutex_t mtx;
 //std::ofstream 
 FILE *fout_kpm;
 char filename[] = "hodata/KPM_UE.txt";
 char mode[] = "a+";
 char mode0[] = "w+";
 int i = 0;
 int num_c_space = 5;
 // int gnb_num = 0;
 // int num_gnb = 3;
 // int gnb_id[3] = {3586, 3585, 3584};
 
 //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@mac_moni@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 
 
 //######################################rb_ctrl#################################
 typedef enum{
     DRX_parameter_configuration_7_6_3_1 = 1,
     SR_periodicity_configuration_7_6_3_1 = 2,
     SPS_parameters_configuration_7_6_3_1 = 3,
     Configured_grant_control_7_6_3_1 = 4,
     CQI_table_configuration_7_6_3_1 = 5,
     Slice_level_PRB_quotal_7_6_3_1 = 6,     //slicing allocation
     Interference_PRB_quotal_7_6_3_1 = 7,    //interference management
     Handover_PRB_quotal_7_6_3_1 = 8,        //handover and resource allocation

 } rc_ctrl_service_style_2_act_id_e;
 
 static
 e2sm_rc_ctrl_hdr_frmt_1_t gen_rc_ctrl_hdr_frmt_1(ue_id_e2sm_t ue_id, uint32_t ric_style_type, uint16_t ctrl_act_id)
 {
   e2sm_rc_ctrl_hdr_frmt_1_t dst = {0};
 
   // 6.2.2.6
   dst.ue_id = cp_ue_id_e2sm(&ue_id);
 
   dst.ric_style_type = ric_style_type;
   dst.ctrl_act_id = ctrl_act_id;
 
   return dst;
 }
 
 static
 e2sm_rc_ctrl_hdr_t gen_rc_ctrl_hdr(e2sm_rc_ctrl_hdr_e hdr_frmt, ue_id_e2sm_t ue_id, uint32_t ric_style_type, uint16_t ctrl_act_id)
 {
   e2sm_rc_ctrl_hdr_t dst = {0};
 
   if (hdr_frmt == FORMAT_1_E2SM_RC_CTRL_HDR) {
     dst.format = FORMAT_1_E2SM_RC_CTRL_HDR;
     dst.frmt_1 = gen_rc_ctrl_hdr_frmt_1(ue_id, ric_style_type, ctrl_act_id);
   } else {
     assert(0!=0 && "not implemented the fill func for this ctrl hdr frmt");
   }
 
   return dst;
 }
 
 typedef enum {
     RRM_Policy_Ratio_List_8_4_3_7 = 1,
     RRM_Policy_Ratio_Group_8_4_3_7 = 2,
     Du_Map_8_4_3_7 = 3,
     //Int_Map_8_4_3_7 = 4,
 } interference_PRB_quota_param_id_e;
 
 static
 void gen_rrm_policy_ratio_group(lst_ran_param_t* RRM_Policy_Ratio_Group,
                                 float du_ratio_prb[num_c_space])
                                 //int num_rbg)
                                 //int a[10])
 {
   // RRM Policy Ratio Group, STRUCTURE (RRM Policy Ratio List -> RRM Policy Ratio Group)
   // lst_ran_param_t* RRM_Policy_Ratio_Group = &RRM_Policy_Ratio_List->ran_param_val.lst->lst_ran_param[0];
   // RRM_Policy_Ratio_Group->ran_param_id = RRM_Policy_Ratio_Group_8_4_3_6;
   RRM_Policy_Ratio_Group->ran_param_struct.sz_ran_param_struct = num_c_space;
   RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct = calloc(num_c_space, sizeof(seq_ran_param_t));
   assert(RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct != NULL && "Memory exhausted");
   // RRM Policy, STRUCTURE (Rgen_rrm_policy_ratio_listRM Policy Ratio Group -> RRM Policy)
   // seq_ran_param_t* RRM_Policy = &RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct[0];
   for (int i = 0; i < num_c_space; i ++){
     seq_ran_param_t* Du_Map = &RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct[i];
     Du_Map->ran_param_id = Du_Map_8_4_3_7;
     Du_Map->ran_param_val.type = ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE;
     Du_Map->ran_param_val.flag_false = calloc(1, sizeof(ran_parameter_value_t));
     assert(Du_Map->ran_param_val.flag_false != NULL && "Memory exhausted");
     Du_Map->ran_param_val.flag_false->type = INTEGER_RAN_PARAMETER_VALUE;
     //Du_Map->ran_param_val.flag_false->int_ran = du_ratio_prb;
     //for (int i = 0; i < 10; i++) {
     Du_Map->ran_param_val.flag_false->real_ran = du_ratio_prb[i];
    //printf("%f\n", Du_Map->ran_param_val.flag_false->real_ran);
     // }
   }
 
 /*
   seq_ran_param_t* int_Map = &RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct[1];
   int_Map->ran_param_id = Int_Map_8_4_3_7;
   int_Map->ran_param_val.type = ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE;
   int_Map->ran_param_val.flag_false = calloc(1, sizeof(ran_parameter_value_t));
   assert(int_Map->ran_param_val.flag_false != NULL && "Memory exhausted");
   int_Map->ran_param_val.flag_false->type = INTEGER_RAN_PARAMETER_VALUE;
   //Du_Map->ran_param_val.flag_false->int_ran = du_ratio_prb;
   int_Map->ran_param_val.flag_false->int_ran = (int *)malloc(10 * sizeof(int));
   for (int i = 0; i < 10; i++) {
     int_Map->ran_param_val.flag_false->int_ran[i] = a[i];
     printf("%d\n", int_Map->ran_param_val.flag_false->int_ran[i]);
   }
 */
   //free(int_Map->ran_param_val.flag_false->int_ran);
 
   return;
 }
 
 static
 void gen_rrm_policy_ratio_list(seq_ran_param_t* RRM_Policy_Ratio_List)
 {
   int a, num_ue, num_gnb;
  
   char line[1024];
 
   FILE *file;
   
   int num[3];

   
    while (1) {
        file = fopen("hodata/ctrl.csv", "r");
        if (file == NULL) {
            usleep(1000); 
            continue;
        }

        
        if (fgets(line, sizeof(line), file)) {
            if (sscanf(line, "%d,", &a) == 1) { 
                if (a == 2) {
                    usleep(1000); 
                    fclose(file); 
                    break;
                }
            }
        }
        
        fclose(file);
        usleep(1000); 
    }

    file = fopen("hodata/ctrl.csv", "r");
    if (file == NULL) {
        return; 
    }

    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return;
    }
    sscanf(line, "%d, %d, %d, %d, %d, %d", &a, &num_ue, &num_gnb, &num[0], &num[1], &num[2]);

    int nodes_id;
    if (nb_id == 3585) {
        nodes_id = 0;
    } else if (nb_id == 3586) {
        nodes_id = 1;
    } else if (nb_id == 3587) {
        nodes_id = 2;
    }else {
        fclose(file);
        return;
    }
    printf("nodes_id = %d\n", nodes_id);

    int du_num_ue = num[nodes_id];
    printf("du_num_ue = %d \n", du_num_ue);

    int first_ue_id = 0;
    for (int i = 0; i < nodes_id; i++) {
        first_ue_id += num[i];
    }

    for (int i = 0; i < first_ue_id; i++) {
        if (!fgets(line, sizeof(line), file)) {
            fclose(file);
            return;
        }
    }

    int num_c_space = 5; 
    float **c = malloc(du_num_ue * sizeof(float *));
    for (int i = 0; i < du_num_ue; i++) {
        c[i] = malloc(num_c_space * sizeof(float));
    }

    for (int i = 0; i < du_num_ue; i++) {
        if (fgets(line, sizeof(line), file)) {
            sscanf(line, "%*d,%*d,\"[%f,%f,%f,%f,%f]\",%*d",
                   &c[i][0], &c[i][1], &c[i][2], &c[i][3], &c[i][4]);
        } else {
            for(int j = 0; j <= i; j++) free(c[j]);
            free(c);
            fclose(file);
            return;
        }
    }
    fclose(file); 

    
   //int num_ue = 2;
   // seq_ran_param_t* RRM_Policy_Ratio_List =  &dst.ran_param[0];
   RRM_Policy_Ratio_List->ran_param_id = RRM_Policy_Ratio_List_8_4_3_7;
   RRM_Policy_Ratio_List->ran_param_val.type = LIST_RAN_PARAMETER_VAL_TYPE;
   RRM_Policy_Ratio_List->ran_param_val.lst = calloc(1, sizeof(ran_param_list_t));
   assert(RRM_Policy_Ratio_List->ran_param_val.lst != NULL && "Memory exhausted");
   RRM_Policy_Ratio_List->ran_param_val.lst->sz_lst_ran_param = du_num_ue;
   RRM_Policy_Ratio_List->ran_param_val.lst->lst_ran_param = calloc(du_num_ue, sizeof(lst_ran_param_t));
   assert(RRM_Policy_Ratio_List->ran_param_val.lst->lst_ran_param != NULL && "Memory exhausted");
 
   int ratio_prb[10];
   for (int i = 0; i < 10; i++){
     ratio_prb[i] = rand() %2;
   }  */
 
   for (int i = 0; i < du_num_ue; i++) {
     gen_rrm_policy_ratio_group(&RRM_Policy_Ratio_List->ran_param_val.lst->lst_ran_param[i],
                                c[i]);
   }
 
   for (int i = 0; i < du_num_ue; i++) {
     free(c[i]);
   }
   free(c);

    FILE *fp_w = fopen("hodata/ctrl.csv", "r+");
    if (fp_w == NULL) {
        return;
    }

    fprintf(fp_w, "0, %d, %d, %d, %d\n", num_ue, num_gnb, num[0], num[1]);
    
    fclose(fp_w);

    usleep(1000);
    
   return;
 }
 
 static
 e2sm_rc_ctrl_msg_frmt_1_t gen_rc_ctrl_msg_frmt_1_slice_level_PRB_quota()
 {
   e2sm_rc_ctrl_msg_frmt_1_t dst = {0};
 
   // RRM Policy Ratio List, LIST
   dst.sz_ran_param = 1;
   dst.ran_param = calloc(1, sizeof(seq_ran_param_t));
   assert(dst.ran_param != NULL && "Memory exhausted");
   gen_rrm_policy_ratio_list(&dst.ran_param[0]);
 
   return dst;
 }
 
 static
 e2sm_rc_ctrl_msg_t gen_rc_ctrl_msg(e2sm_rc_ctrl_msg_e msg_frmt)
 {
   e2sm_rc_ctrl_msg_t dst = {0};
 
   if (msg_frmt == FORMAT_1_E2SM_RC_CTRL_MSG) {
     dst.format = msg_frmt;
     dst.frmt_1 = gen_rc_ctrl_msg_frmt_1_slice_level_PRB_quota();
   } else {
     assert(0!=0 && "not implemented the fill func for this ctrl msg frmt");
   }
 
   return dst;
 }
 
 static
 ue_id_e2sm_t gen_rc_ue_id(ue_id_e2sm_e type)
 {
   ue_id_e2sm_t ue_id = {0};
   if (type == GNB_UE_ID_E2SM) {
     ue_id.type = GNB_UE_ID_E2SM;
     // TODO
     ue_id.gnb.amf_ue_ngap_id = 0;
     ue_id.gnb.guami.plmn_id.mcc = 1;
     ue_id.gnb.guami.plmn_id.mnc = 1;
     ue_id.gnb.guami.plmn_id.mnc_digit_len = 2;
     ue_id.gnb.guami.amf_region_id = 0;
     ue_id.gnb.guami.amf_set_id = 0;
     ue_id.gnb.guami.amf_ptr = 0;
   } else {
     assert(0!=0 && "not supported UE ID type");
   }
   return ue_id;
 }
 
 //***************************************kpm_moni********************************
 static
 void log_gnb_ue_id(ue_id_e2sm_t ue_id)
 {
   /*
   i++;
   printf("###############the totle number of lines in the file%d", i);
   if (i>2){
     fout_kpm = fopen(filename, mode0);
     i = 0;
   }else{
     fout_kpm = fopen(filename, mode);
   }*/
   fout_kpm = fopen(filename, mode);
   if (ue_id.gnb.gnb_cu_ue_f1ap_lst != NULL) {
     for (size_t i = 0; i < ue_id.gnb.gnb_cu_ue_f1ap_lst_len; i++) {
       //printf("UE ID type = gNB-CU, gnb_cu_ue_f1ap = %u\n", ue_id.gnb.gnb_cu_ue_f1ap_lst[i]);
     }
   } else {
     //printf("UE ID type = gNB, amf_ue_ngap_id = %lu\n", ue_id.gnb.amf_ue_ngap_id);
   }
   if (ue_id.gnb.ran_ue_id != NULL) {
     // printf("DU_ID = %d\n", gnb_id[gnb_num-1]); // RAN UE NGAP ID
     // fseek(fout_kpm, 0, SEEK_SET);
     // fprintf(fout_kpm, "%d ", gnb_id[gnb_num-1]);
 
     //printf("ran_ue_id = %lx\n", *ue_id.gnb.ran_ue_id); // RAN UE NGAP ID
     fseek(fout_kpm, 0, SEEK_SET);
     fprintf(fout_kpm, "%ld ", *ue_id.gnb.ran_ue_id);
     fclose(fout_kpm);
   }
 }
 
 static
 void log_du_ue_id(ue_id_e2sm_t ue_id)
 {
   /*
   i++;
   printf("###############the totle number of lines in the file%d", i);
   if (i>2){
     fout_kpm = fopen(filename, mode0);
     i = 0;
   }else{
     fout_kpm = fopen(filename, mode);
   }
   */
   fout_kpm = fopen(filename, mode);
   //printf("UE ID type = gNB-DU, gnb_cu_ue_f1ap = %u\n", ue_id.gnb_du.gnb_cu_ue_f1ap);
   if (ue_id.gnb_du.ran_ue_id != NULL) {
     
     // printf("DU_ID = %d\n", gnb_id[gnb_num-1]); // RAN UE NGAP ID
     // fseek(fout_kpm, 0, SEEK_SET);
     // fprintf(fout_kpm, "%d ", gnb_id[gnb_num-1]);
 
     //printf("ran_ue_id = %lx\n", *ue_id.gnb_du.ran_ue_id); // RAN UE NGAP ID
     fseek(fout_kpm, 0, SEEK_SET);
     fprintf(fout_kpm, "%ld ", *ue_id.gnb_du.ran_ue_id);
     
     fclose(fout_kpm);
   }
 }
 
 static
 void log_cuup_ue_id(ue_id_e2sm_t ue_id)
 {
   //printf("UE ID type = gNB-CU-UP, gnb_cu_cp_ue_e1ap = %u\n", ue_id.gnb_cu_up.gnb_cu_cp_ue_e1ap);
   if (ue_id.gnb_cu_up.ran_ue_id != NULL) {
     //printf("ran_ue_id = %lx\n", *ue_id.gnb_cu_up.ran_ue_id); // RAN UE NGAP ID
   }
 }
 
 typedef void (*log_ue_id)(ue_id_e2sm_t ue_id);
 
 static
 log_ue_id log_ue_id_e2sm[END_UE_ID_E2SM] = {
     log_gnb_ue_id, // common for gNB-mono, CU and CU-CP
     log_du_ue_id,
     log_cuup_ue_id,
     NULL,
     NULL,
     NULL,
     NULL,
 };
 
 
 
 static
 void log_int_value(byte_array_t name, meas_record_lst_t meas_record)
 {
   fout_kpm = fopen(filename, mode);
   if (cmp_str_ba("RRU.PrbTotDl", name) == 0) {
     //printf("RRU.PrbTotDl = %d [PRBs]\n", meas_record.int_val);
     fseek(fout_kpm, 0, SEEK_SET);
     fprintf(fout_kpm, "%d ", meas_record.int_val);
     fclose(fout_kpm);
   } else if (cmp_str_ba("RRU.PrbTotUl", name) == 0) {
     //printf("RRU.PrbTotUl = %d [PRBs]\n", meas_record.int_val);
     fseek(fout_kpm, 0, SEEK_SET);
     fprintf(fout_kpm, "%d\n", meas_record.int_val);
     fclose(fout_kpm);
   } else if (cmp_str_ba("DRB.PdcpSduVolumeDL", name) == 0) {
     //printf("DRB.PdcpSduVolumeDL = %d [kb]\n", meas_record.int_val);
     fseek(fout_kpm, 0, SEEK_SET);
     fprintf(fout_kpm, "%d ", meas_record.int_val);
     fclose(fout_kpm);
   } else if (cmp_str_ba("DRB.PdcpSduVolumeUL", name) == 0) {
     //printf("DRB.PdcpSduVolumeUL = %d [kb]\n", meas_record.int_val);
     fseek(fout_kpm, 0, SEEK_SET);
     fprintf(fout_kpm, "%d ", meas_record.int_val);
     fclose(fout_kpm);
   } else {
     printf("Measurement Name not yet supported\n");
   }
 }
 
 static
 void log_real_value(byte_array_t name, meas_record_lst_t meas_record)
 {
   fout_kpm = fopen(filename, mode);
   if (cmp_str_ba("DRB.RlcSduDelayDl", name) == 0) {
     //printf("DRB.RlcSduDelayDl = %.2f [μs]\n", meas_record.real_val);
     fseek(fout_kpm, 0, SEEK_SET);
     fprintf(fout_kpm, "%.2f ", meas_record.real_val);
     fclose(fout_kpm);
   } else if (cmp_str_ba("DRB.UEThpDl", name) == 0) {
     //printf("DRB.UEThpDl = %.2f [kbps]\n", meas_record.real_val);
     fseek(fout_kpm, 0, SEEK_SET);
     fprintf(fout_kpm, "%.2f ", meas_record.real_val);
     fclose(fout_kpm);
   } else if (cmp_str_ba("DRB.UEThpUl", name) == 0) {
     //printf("DRB.UEThpUl = %.2f [kbps]\n", meas_record.real_val);
     fseek(fout_kpm, 0, SEEK_SET);
     fprintf(fout_kpm, "%.2f ", meas_record.real_val);
     fclose(fout_kpm);
   } else {
     //printf("Measurement Name not yet supported\n");
   }
 }
 
 typedef void (*log_meas_value)(byte_array_t name, meas_record_lst_t meas_record);
 
 static
 log_meas_value get_meas_value[END_MEAS_VALUE] = {
     log_int_value,
     log_real_value,
     NULL,
 };
 
 static
 void match_meas_name_type(meas_type_t meas_type, meas_record_lst_t meas_record)
 {
   // Get the value of the Measurement
   get_meas_value[meas_record.value](meas_type.name, meas_record);
 }
 
 static
 void match_id_meas_type(meas_type_t meas_type, meas_record_lst_t meas_record)
 {
   (void)meas_type;
   (void)meas_record;
   assert(false && "ID Measurement Type not yet supported");
 }
 
 typedef void (*check_meas_type)(meas_type_t meas_type, meas_record_lst_t meas_record);
 
 static
 check_meas_type match_meas_type[END_MEAS_TYPE] = {
     match_meas_name_type,
     match_id_meas_type,
 };
 
 static
 void log_kpm_measurements(kpm_ind_msg_format_1_t const* msg_frm_1)
 {
   assert(msg_frm_1->meas_info_lst_len > 0 && "Cannot correctly print measurements");
 
   // UE Measurements per granularity period
   for (size_t j = 0; j < msg_frm_1->meas_data_lst_len; j++) {
     meas_data_lst_t const data_item = msg_frm_1->meas_data_lst[j];
 
     for (size_t z = 0; z < data_item.meas_record_len; z++) {
       meas_type_t const meas_type = msg_frm_1->meas_info_lst[z].meas_type;
       meas_record_lst_t const record_item = data_item.meas_record_lst[z];
 
       match_meas_type[meas_type.type](meas_type, record_item);
 
       if (data_item.incomplete_flag && *data_item.incomplete_flag == TRUE_ENUM_VALUE)
         printf("Measurement Record not reliable");
     }
   }
 
 }
 
 static
 void sm_cb_kpm(sm_ag_if_rd_t const* rd)   //, global_e2_node_id_t* id
 {
   assert(rd != NULL);
   assert(rd->type == INDICATION_MSG_AGENT_IF_ANS_V0);
   assert(rd->ind.type == KPM_STATS_V3_0);
 
   // Reading Indication Message Format 3
   kpm_ind_data_t const* ind = &rd->ind.kpm.ind;
   kpm_ric_ind_hdr_format_1_t const* hdr_frm_1 = &ind->hdr.kpm_ric_ind_hdr_format_1;
   kpm_ind_msg_format_3_t const* msg_frm_3 = &ind->msg.frm_3;
 
   int64_t const now = time_now_us();
   int num_cell = 6;
   static int counter = 1;                                                                                
   {
     lock_guard(&mtx);
     // if(gnb_num == num_gnb) gnb_num = 0;
     // gnb_num = gnb_num + 1;
     //printf("\n%d \n", gnb_num);

     // printf("\n%7d KPM ind_msg latency = %ld [μs]\n", counter, now - hdr_frm_1->collectStartTime); // xApp <-> E2 Node   //here is call back for one time
    
     //  printf("du_id = %d \n", id->nb_id.nb_id);
     // Reported list of measurements per UE
     if (counter % num_cell == 0){
      fout_kpm = fopen(filename, mode0);
      fclose(fout_kpm);
     }
     for (size_t i = 0; i < msg_frm_3->ue_meas_report_lst_len; i++) {
       // log UE ID
       ue_id_e2sm_t const ue_id_e2sm = msg_frm_3->meas_report_per_ue[i].ue_meas_report_lst;
       ue_id_e2sm_e const type = ue_id_e2sm.type;
       log_ue_id_e2sm[type](ue_id_e2sm);
 
       // log measurements
       log_kpm_measurements(&msg_frm_3->meas_report_per_ue[i].ind_msg_format_1);
       
     }
     counter++;
   }
 }
 
 static
 test_info_lst_t filter_predicate(test_cond_type_e type, test_cond_e cond, int value)
 {
   test_info_lst_t dst = {0};
 
   dst.test_cond_type = type;
   // It can only be TRUE_TEST_COND_TYPE so it does not matter the type
   // but ugly ugly...
   dst.S_NSSAI = TRUE_TEST_COND_TYPE;
 
   dst.test_cond = calloc(1, sizeof(test_cond_e));
   assert(dst.test_cond != NULL && "Memory exhausted");
   *dst.test_cond = cond;
 
   dst.test_cond_value = calloc(1, sizeof(test_cond_value_t));
   assert(dst.test_cond_value != NULL && "Memory exhausted");
   dst.test_cond_value->type = OCTET_STRING_TEST_COND_VALUE;
 
   dst.test_cond_value->octet_string_value = calloc(1, sizeof(byte_array_t));
   assert(dst.test_cond_value->octet_string_value != NULL && "Memory exhausted");
   const size_t len_nssai = 1;
   dst.test_cond_value->octet_string_value->len = len_nssai;
   dst.test_cond_value->octet_string_value->buf = calloc(len_nssai, sizeof(uint8_t));
   assert(dst.test_cond_value->octet_string_value->buf != NULL && "Memory exhausted");
   dst.test_cond_value->octet_string_value->buf[0] = value;
 
   return dst;
 }
 
 static
 label_info_lst_t fill_kpm_label(void)
 {
   label_info_lst_t label_item = {0};
 
   label_item.noLabel = calloc(1, sizeof(enum_value_e));
   *label_item.noLabel = TRUE_ENUM_VALUE;
 
   return label_item;
 }
 
 static
 kpm_act_def_format_1_t fill_act_def_frm_1(ric_report_style_item_t const* report_item)
 {
   assert(report_item != NULL);
 
   kpm_act_def_format_1_t ad_frm_1 = {0};
 
   size_t const sz = report_item->meas_info_for_action_lst_len;
 
   // [1, 65535]
   ad_frm_1.meas_info_lst_len = sz;
   ad_frm_1.meas_info_lst = calloc(sz, sizeof(meas_info_format_1_lst_t));
   assert(ad_frm_1.meas_info_lst != NULL && "Memory exhausted");
 
   for (size_t i = 0; i < sz; i++) {
     meas_info_format_1_lst_t* meas_item = &ad_frm_1.meas_info_lst[i];
     // 8.3.9
     // Measurement Namecreate_mac_ue_table
     meas_item->meas_type.type = NAME_MEAS_TYPE;
     meas_item->meas_type.name = copy_byte_array(report_item->meas_info_for_action_lst[i].name);
 
     // [1, 2147483647]
     // 8.3.11
     meas_item->label_info_lst_len = 1;
     meas_item->label_info_lst = calloc(1, sizeof(label_info_lst_t));
     meas_item->label_info_lst[0] = fill_kpm_label();
   }
 
   // 8.3.8 [0, 4294967295]
   ad_frm_1.gran_period_ms = period_ms;
 
   // 8.3.20 - OPTIONAL
   ad_frm_1.cell_global_id = NULL;
 
 #if defined KPM_V2_03 || defined KPM_V3_00
   // [0, 65535]
   ad_frm_1.meas_bin_range_info_lst_len = 0;
   ad_frm_1.meas_bin_info_lst = NULL;
 #endif
 
   return ad_frm_1;
 }
 
 static
 kpm_act_def_t fill_report_style_4(ric_report_style_item_t const* report_item)
 {
   assert(report_item != NULL);
   assert(report_item->act_def_format_type == FORMAT_4_ACTION_DEFINITION);
 
   kpm_act_def_t act_def = {.type = FORMAT_4_ACTION_DEFINITION};
 
   // Fill matching condition
   // [1, 32768]
   act_def.frm_4.matching_cond_lst_len = 1;
   act_def.frm_4.matching_cond_lst = calloc(act_def.frm_4.matching_cond_lst_len, sizeof(matching_condition_format_4_lst_t));
   assert(act_def.frm_4.matching_cond_lst != NULL && "Memory exhausted");
   // Filter connected UEs by S-NSSAI criteria
   test_cond_type_e const type = S_NSSAI_TEST_COND_TYPE; // CQI_TEST_COND_TYPE
   test_cond_e const condition = EQUAL_TEST_COND; // GREATERTHAN_TEST_COND
   int const value = 1;
   act_def.frm_4.matching_cond_lst[0].test_info_lst = filter_predicate(type, condition, value);
 
   // Fill Action Definition Format 1
   // 8.2.1.2.1
   act_def.frm_4.action_def_format_1 = fill_act_def_frm_1(report_item);
 
   return act_def;
 }
 
 typedef kpm_act_def_t (*fill_kpm_act_def)(ric_report_style_item_t const* report_item);
 
 static
 fill_kpm_act_def get_kpm_act_def[END_RIC_SERVICE_REPORT] = {
     NULL,
     NULL,
     NULL,
     fill_report_style_4,
     NULL,
 };
 
 static
 kpm_sub_data_t gen_kpm_subs(kpm_ran_function_def_t const* ran_func)
 {
   assert(ran_func != NULL);
   assert(ran_func->ric_event_trigger_style_list != NULL);
 
   kpm_sub_data_t kpm_sub = {0};
 
   // Generate Event Trigger
   assert(ran_func->ric_event_trigger_style_list[0].format_type == FORMAT_1_RIC_EVENT_TRIGGER);
   kpm_sub.ev_trg_def.type = FORMAT_1_RIC_EVENT_TRIGGER;
   kpm_sub.ev_trg_def.kpm_ric_event_trigger_format_1.report_period_ms = period_ms;
 
   // Generate Action Definition
   kpm_sub.sz_ad = 1;
   kpm_sub.ad = calloc(kpm_sub.sz_ad, sizeof(kpm_act_def_t));
   assert(kpm_sub.ad != NULL && "Memory exhausted");
 
   // Multiple Action Definitions in one SUBSCRIPTION message is not supported in this project
   // Multiple REPORT Styles = Multiple Action Definition = Multiple SUBSCRIPTION messages
   ric_report_style_item_t* const report_item = &ran_func->ric_report_style_list[0];
   ric_service_report_e const report_style_type = report_item->report_style_type;
   *kpm_sub.ad = get_kpm_act_def[report_style_type](report_item);
 
   return kpm_sub;
 }
 
 static
 bool eq_sm(sm_ran_function_t const* elem, int const id)
 {
   if (elem->id == id)
     return true;
 
   return false;
 }
 
 static
 size_t find_sm_idx(sm_ran_function_t* rf, size_t sz, bool (*f)(sm_ran_function_t const*, int const), int const id)
 {
   for (size_t i = 0; i < sz; i++) {
     if (f(&rf[i], id))
       return i;
   }
 
   assert(0 != 0 && "SM ID could not be found in the RAN Function List");
 }
 
 //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@mac_moni@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 static
 uint64_t cnt_mac;
 
 static
 void sm_cb_mac(sm_ag_if_rd_t const* rd)
 {
   assert(rd != NULL);
   assert(rd->type ==INDICATION_MSG_AGENT_IF_ANS_V0);
   assert(rd->ind.type == MAC_STATS_V0);
  
   int64_t now = time_now_us();
   if(cnt_mac % 1024 == 0)
     printf("MAC ind_msg latency = %ld μs\n", now - rd->ind.mac.msg.tstamp);
   cnt_mac++;
 }
 
 static
 uint64_t cnt_rlc;
 
 static
 void sm_cb_rlc(sm_ag_if_rd_t const* rd)
 {
   assert(rd != NULL);
   assert(rd->type ==INDICATION_MSG_AGENT_IF_ANS_V0);
 
   assert(rd->ind.type == RLC_STATS_V0);
 
   int64_t now = time_now_us();
 
   if(cnt_rlc % 1024 == 0)
     printf("RLC ind_msg latency = %ld μs\n", now - rd->ind.rlc.msg.tstamp);
   cnt_rlc++;
 }
 
 static
 uint64_t cnt_pdcp;
 
 static
 void sm_cb_pdcp(sm_ag_if_rd_t const* rd)
 {
   assert(rd != NULL);
   assert(rd->type == INDICATION_MSG_AGENT_IF_ANS_V0);
 
   assert(rd->ind.type == PDCP_STATS_V0);
 
   int64_t now = time_now_us();
 
   if(cnt_pdcp % 1024 == 0)
     printf("PDCP ind_msg latency = %ld μs\n", now - rd->ind.pdcp.msg.tstamp);
 
   cnt_pdcp++;
 }
 
 static
 uint64_t cnt_gtp;
 
 static
 void sm_cb_gtp(sm_ag_if_rd_t const* rd)
 {
   assert(rd != NULL);
   assert(rd->type ==INDICATION_MSG_AGENT_IF_ANS_V0);
 
   assert(rd->ind.type == GTP_STATS_V0);
 
   int64_t now = time_now_us();
   if(cnt_gtp % 1024 == 0)
     printf("GTP ind_msg latency = %ld μs\n", now - rd->ind.gtp.msg.tstamp);
 
   cnt_gtp++;
 }
 
 
 
 
 int main(int argc, char *argv[])
 {
   fr_args_t args = init_fr_args(argc, argv);
   //defer({ free_fr_args(&args); });
 
   //Init the xApp
   init_xapp_api(&args);
   sleep(1);
 
   e2_node_arr_xapp_t nodes = e2_nodes_xapp_api();
   defer({ free_e2_node_arr_xapp(&nodes); });
   assert(nodes.len > 0);
   printf("Connected E2 nodes = %d\n", nodes.len);
 
   //***************
   // START KPM Moni
   //***************
 
 pthread_mutexattr_t attr = {0};
   int rc = pthread_mutex_init(&mtx, &attr);
   assert(rc == 0);
 
   sm_ans_xapp_t* hndl = calloc(nodes.len, sizeof(sm_ans_xapp_t));
   assert(hndl != NULL);
 
   ////////////
   // START KPM
   ////////////
   int const KPM_ran_function = 2;
 
   fout_kpm = fopen(filename, mode0);
   if (fout_kpm == NULL) { 
     printf("cannot open the file\n");
     return 1;
   }
   printf("Successfull open the file\n");
 
   for (size_t i = 0; i < nodes.len; ++i) {
     e2_node_connected_xapp_t* n = &nodes.n[i];
 
     size_t const idx = find_sm_idx(n->rf, n->len_rf, eq_sm, KPM_ran_function);
     assert(n->rf[idx].defn.type == KPM_RAN_FUNC_DEF_E && "KPM is not the received RAN Function");
     // if REPORT Service is supported by E2 node, send SUBSCRIPTION
     // e.g. OAI CU-CP
     if (n->rf[idx].defn.kpm.ric_report_style_list != NULL) {
       // Generate KPM SUBSCRIPTION message
       kpm_sub_data_t kpm_sub = gen_kpm_subs(&n->rf[idx].defn.kpm);
 
       hndl[i] = report_sm_xapp_api(&n->id, KPM_ran_function, &kpm_sub, sm_cb_kpm);
       printf("du_id = %d \n", n->id.nb_id.nb_id);
       assert(hndl[i].success == true);
 
       free_kpm_sub_data(&kpm_sub);
     }
   }
   
   //@@@@@@@@@@@@
   // START MAC
   //@@@@@@@@@@@@
 
 
 // MAC indication
   const char* i_0 = "10_ms";
   sm_ans_xapp_t* mac_handle = NULL;
  //  RLC indication
   const char* i_1 = "10_ms";
   sm_ans_xapp_t* rlc_handle = NULL;
   // // PDCP indication
   // const char* i_2 = "10_ms";
   // sm_ans_xapp_t* pdcp_handle = NULL;
   // // GTP indication
   // const char* i_3 = "10_ms";
   // sm_ans_xapp_t* gtp_handle = NULL;
 
 
   if(nodes.len > 0){
     mac_handle = calloc( nodes.len, sizeof(sm_ans_xapp_t) ); 
     assert(mac_handle  != NULL);
     rlc_handle = calloc( nodes.len, sizeof(sm_ans_xapp_t) ); 
     assert(rlc_handle  != NULL);
     // pdcp_handle = calloc( nodes.len, sizeof(sm_ans_xapp_t) ); 
     // assert(pdcp_handle  != NULL);
     // gtp_handle = calloc( nodes.len, sizeof(sm_ans_xapp_t) ); 
     // assert(gtp_handle  != NULL);
   }
 
   for (int i = 0; i < nodes.len; i++) {
     e2_node_connected_xapp_t* n = &nodes.n[i];
     for (size_t j = 0; j < n->len_rf; j++)
       printf("Registered node %d ran func id = %d \n ", i, n->rf[j].id);
 
     if(n->id.type == ngran_gNB || n->id.type == ngran_eNB){
       // MAC Control is not yet implemented in OAI RAN
       // mac_ctrl_req_data_t wr = {.hdr.dummy = 1, .msg.action = 42 };
       // sm_ans_xapp_t const a = control_sm_xapp_api(&nodes.n[i].id, 142, &wr);
       // assert(a.success == true);
 
       mac_handle[i] = report_sm_xapp_api(&nodes.n[i].id, 142, (void*)i_0, sm_cb_mac);
       assert(mac_handle[i].success == true);
 
       rlc_handle[i] = report_sm_xapp_api(&nodes.n[i].id, 143, (void*)i_1, sm_cb_rlc);
       assert(rlc_handle[i].success == true);
 
     //   pdcp_handle[i] = report_sm_xapp_api(&nodes.n[i].id, 144, (void*)i_2, sm_cb_pdcp);
     //   assert(pdcp_handle[i].success == true);
 
     //   gtp_handle[i] = report_sm_xapp_api(&nodes.n[i].id, 148, (void*)i_3, sm_cb_gtp);
     //   assert(gtp_handle[i].success == true);
 
     // } else if(n->id.type ==  ngran_gNB_CU || n->id.type ==  ngran_gNB_CUUP){
     //   pdcp_handle[i] = report_sm_xapp_api(&nodes.n[i].id, 144, (void*)i_2, sm_cb_pdcp);
     //   assert(pdcp_handle[i].success == true);
 
     //   gtp_handle[i] = report_sm_xapp_api(&nodes.n[i].id, 148, (void*)i_3, sm_cb_gtp);
     //   assert(gtp_handle[i].success == true);
 
     } else if(n->id.type == ngran_gNB_DU){
       mac_handle[i] = report_sm_xapp_api(&nodes.n[i].id, 142, (void*)i_0, sm_cb_mac);
       assert(mac_handle[i].success == true);
 
       rlc_handle[i] = report_sm_xapp_api(&nodes.n[i].id, 143, (void*)i_1, sm_cb_rlc);
       assert(rlc_handle[i].success == true);
     }
 
   }
 
   ////////////
   // START RC
   ////////////
 
   // RC Control
   // CONTROL Service Style 2: Radio Resource Allocation Control
   // Action ID 6: Slice-level PRB quota
   // E2SM-RC Control Header Format 1
   // E2SM-RC Control Message Format 1
  
   rc_ctrl_req_data_t rc_ctrl = {0};
   ue_id_e2sm_t ue_id = gen_rc_ue_id(GNB_UE_ID_E2SM);
 
   rc_ctrl.hdr = gen_rc_ctrl_hdr(FORMAT_1_E2SM_RC_CTRL_HDR, ue_id, 3, Handover_PRB_quotal_7_6_3_1);
   //c_ctrl.msg = gen_rc_ctrl_msg(FORMAT_1_E2SM_RC_CTRL_MSG);
 
   int64_t st = time_now_us();
 
   while(1){
     for(size_t i = 0; i < nodes.len; ++i){
       nb_id = nodes.n[i].id.nb_id.nb_id;
       rc_ctrl.msg = gen_rc_ctrl_msg(FORMAT_1_E2SM_RC_CTRL_MSG);
       control_sm_xapp_api(&nodes.n[i].id, SM_RC_ID, &rc_ctrl);
     }
     printf("[xApp]: Control Loop Latency: %ld us\n", time_now_us() - st);
     free_rc_ctrl_req_data(&rc_ctrl);
     usleep(1000);
   }
   
 
   //***************
   // END KPM Moni
   //***************
 
   for (int i = 0; i < nodes.len; ++i) {
     // Remove the handle previously returned
     if (hndl[i].success == true)
       rm_report_sm_xapp_api(hndl[i].u.handle);
   }
   free(hndl);
 
   //@@@@@@@@@@@@
   // END MAC Moni
   //@@@@@@@@@@@@
 
   for(int i = 0; i < nodes.len; ++i){
     // Remove the handle previously returned
     if(mac_handle[i].u.handle != 0 )
      rm_report_sm_xapp_api(mac_handle[i].u.handle);
     if(rlc_handle[i].u.handle != 0) 
      rm_report_sm_xapp_api(rlc_handle[i].u.handle);
     // if(pdcp_handle[i].u.handle != 0)
     //   rm_report_sm_xapp_api(pdcp_handle[i].u.handle);
     // if(gtp_handle[i].u.handle != 0)
     //   rm_report_sm_xapp_api(gtp_handle[i].u.handle);
   }
 
   if(nodes.len > 0){
     free(mac_handle);
     // free(rlc_handle);
     // free(pdcp_handle);
     // free(gtp_handle);
   }
 
   printf("Test xApp run SUCCESSFULLY\n");
 
   ////////////
   // END RC
   ////////////
 
 
   //Stop the xApp
   while(try_stop_xapp_api() == false)
     usleep(1000);
 
   printf("Test xApp run SUCCESSFULLY\n");
 
 }