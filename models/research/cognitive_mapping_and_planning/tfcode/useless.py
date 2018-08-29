#rec['ego_goal_imgs_0']=f['ego_goal_imgs_0'].tolist()
      #rec['ego_goal_imgs_1']=f['ego_goal_imgs_1'].tolist()
      #rec['ego_goal_imgs_2']=f['ego_goal_imgs_2'].tolist()
      '''goal_img_0_pre = np.sum(f['ego_goal_imgs_0'][0, 0,:, :, :], 2)[:,:,np.newaxis]*255.0
      goal_img_0 = np.concatenate((goal_img_0_pre, goal_img_0_pre, goal_img_0_pre), 2)
      goal_img_1_pre = np.sum(f['ego_goal_imgs_1'][0, 0,:, :, :], 2)[:,:,np.newaxis]*255.0
      goal_img_1 = np.concatenate((goal_img_1_pre, goal_img_1_pre, goal_img_1_pre), 2)
      goal_img_2_pre = np.sum(f['ego_goal_imgs_2'][0, 0,:, :, :], 2)[:,:,np.newaxis]*255.0
      goal_img_2 = np.concatenate((goal_img_2_pre, goal_img_2_pre, goal_img_2_pre), 2)
      Image.fromarray(goal_img_0.tolist()).save(logdir+"/logfiles/"+str(n_step)+"/"+str(j)+"_goal_img_0.jpg")
      Image.fromarray(goal_img_1.tolist()).save(logdir+"/logfiles/"+str(n_step)+"/"+str(j)+"_goal_img_1.jpg")
      Image.fromarray(goal_img_2.tolist()).save(logdir+"/logfiles/"+str(n_step)+"/"+str(j)+"_goal_img_2.jpg")
      '''
      '''Image.fromarray(np.uint8(goal_img_0)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_goal_img_0.jpg")
      Image.fromarray(np.uint8(goal_img_1)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_goal_img_1.jpg")
      Image.fromarray(np.uint8(goal_img_2)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_goal_img_2.jpg")
      
      sum_num_0 = net_state['running_sum_num_0'][0, 0, :, :, :3] + 128.0
      sum_num_1 = net_state['running_sum_num_1'][0, 0, :, :, :3] + 128.0
      sum_num_2 = net_state['running_sum_num_2'][0, 0, :, :, :3] + 128.0

      Image.fromarray(np.uint8(sum_num_0)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_sum_num_0.jpg")
      Image.fromarray(np.uint8(sum_num_1)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_sum_num_1.jpg")
      Image.fromarray(np.uint8(sum_num_2)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_sum_num_2.jpg")
      output_file.write(str(f['loc_on_map']))
      output_file.write(str(ori))
      output_file.write(str(f['incremental_locs']))
      output_file.write(str(f['incremental_thetas']))'''
      #f = e.pre_features(f)





#fr_0_pre = np.amax(outs[9][0,  :, :, :], 2)[:,:,np.newaxis]*50.0
      #fr_0 = np.concatenate((fr_0_pre, fr_0_pre, fr_0_pre), 2)
      #fr_1_pre = np.amax(outs[10][0,  :, :, :], 2)[:,:,np.newaxis]*50.0
      #fr_1 = np.concatenate((fr_1_pre, fr_1_pre, fr_1_pre), 2)
      #fr_2_pre = np.amax(outs[11][0,  :, :, :], 2)[:,:,np.newaxis]*50.0
      #fr_2 = np.concatenate((fr_2_pre, fr_2_pre, fr_2_pre), 2)

      #Image.fromarray(np.uint8(fr_0)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_fr_0.jpg")
      #Image.fromarray(np.uint8(fr_1)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_fr_1.jpg")
      #Image.fromarray(np.uint8(fr_2)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_fr_2.jpg")
      

      #vin_0 = outs[6][0,  :, :, :] * 20.0
      #vin_1 = outs[7][0,  :, :, :] * 20.0
      #vin_2 = outs[8][0,  :, :, :] * 20.0

      #print outs[9][0,  :, :, :]
      #print outs[6][0,  :, :, :]

      #Image.fromarray(np.uint8(vin_0)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_vin_0.jpg")
      #Image.fromarray(np.uint8(vin_1)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_vin_1.jpg")
      #Image.fromarray(np.uint8(vin_2)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_vin_2.jpg")
      
      #print (outs[6])
      # outs
      #fpkl.append(rec)






#print(outss)
      '''if np.mod(n_step, 20) == 0:
        for j in range(num_steps):
          ego_img_0_pre = outss[1][0, j,  :, :, 0]
          ego_img_0_pre = ego_img_0_pre[:,:,np.newaxis] * 255.0
          ego_img_0 = np.concatenate((ego_img_0_pre, ego_img_0_pre, ego_img_0_pre), 2)
          
          ego_img_1_pre = outss[1][0, j,  :, :, 1]
          ego_img_1_pre = ego_img_1_pre[:,:,np.newaxis] * 255.0
          ego_img_1 = np.concatenate((ego_img_1_pre, ego_img_1_pre, ego_img_1_pre), 2)
          
          ego_img_2_pre = outss[1][0, j,  :, :, 2]
          ego_img_2_pre = ego_img_2_pre[:,:,np.newaxis] * 255.0
          ego_img_2 = np.concatenate((ego_img_2_pre, ego_img_2_pre, ego_img_2_pre), 2)
          
          Image.fromarray(np.uint8(ego_img_0)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_ego_0.jpg")
          Image.fromarray(np.uint8(ego_img_1)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_ego_1.jpg")
          Image.fromarray(np.uint8(ego_img_2)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_ego_2.jpg")
          
          ego_img_0_pre = outss[2][0, j,  :, :, 0]
          ego_img_0_pre = ego_img_0_pre[:,:,np.newaxis] * 255.0
          ego_img_0 = np.concatenate((ego_img_0_pre, ego_img_0_pre, ego_img_0_pre), 2)
          
          ego_img_1_pre = outss[2][0, j,  :, :, 1]
          ego_img_1_pre = ego_img_1_pre[:,:,np.newaxis] * 255.0
          ego_img_1 = np.concatenate((ego_img_1_pre, ego_img_1_pre, ego_img_1_pre), 2)
          
          ego_img_2_pre = outss[2][0, j,  :, :, 2]
          ego_img_2_pre = ego_img_2_pre[:,:,np.newaxis] * 255.0
          ego_img_2 = np.concatenate((ego_img_2_pre, ego_img_2_pre, ego_img_2_pre), 2)
          
          Image.fromarray(np.uint8(ego_img_0)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_ego_pre_0.jpg")
          Image.fromarray(np.uint8(ego_img_1)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_ego_pre_1.jpg")
          Image.fromarray(np.uint8(ego_img_2)).save(prefix_logdir+str(n_step)+"/"+str(j)+"_ego_pre_2.jpg")
      '''    