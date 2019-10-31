/*
 * Vulkan Example - Texture loading (and display) example (including mip maps)
 *
 * Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gli/gli.hpp>

#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VulkanDevice.hpp"
#include "VulkanBuffer.hpp"

#include "Rpr/RadeonProRender.h"
#include "Rpr/RadeonProRender_Baikal.h"
#include "Rpr/RadeonProRender_VK.h"
#include "Rpr/RadeonProRenderIO.h"

#include "Rpr/Math/mathutils.h"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION true

#ifndef WIN32
    #define PLUGIN_NAME "../libs/Rpr/Hybrid.so"
#else
    #define PLUGIN_NAME "../bin/Hybrid.dll"
#endif

#define CHECK_RPR(x) \
{ \
    rpr_int status = (x); \
    if (status != RPR_SUCCESS)    \
    { \
        std::cerr << "Error: " #x " == " << (status) << ")\n"; \
        assert(false);                      \
    }\
}

class VulkanExample : public VulkanExampleBase
{
public:
    // Contains all Vulkan objects that are required to store and use a texture
    // Note that this repository contains a texture class (VulkanTexture.hpp) that encapsulates texture loading functionality in a class that is used in subsequent demos
    struct Texture {
        VkSampler sampler;
        VkImage image;
        VkImageLayout imageLayout;
        VkDeviceMemory deviceMemory;
        VkImageView view;
        uint32_t width, height;
        uint32_t mipLevels;
    } texture;

    struct {
        VkPipelineVertexInputStateCreateInfo inputState;
        std::vector<VkVertexInputBindingDescription> bindingDescriptions;
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
    } vertices;

    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount;

    struct {
        VkPipeline solid;
    } pipelines;

    VkPipelineLayout pipelineLayout;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    //Interop
    constexpr static std::uint32_t frames_in_flight_ = 3;
    std::uint32_t acc_size_ = 2 * 1024u * 1024u;
    std::uint32_t vbuf_size_ = 1 * 1024u * 1024u;
    std::uint32_t ibuf_size_ = 1 * 1024u * 1024u;
    std::uint32_t sbuf_size_ = 5 * 1024u * 1024u;

    std::array<VkSemaphore, frames_in_flight_> framebuffer_release_semaphores_;
    std::array<VkSemaphore, frames_in_flight_> framebuffer_ready_semaphores_;

    rprContextFlushFrameBuffers_func rprContextFlushFrameBuffers;

    // RPR
    rpr_context context_;
    rpr_material_system mat_system_;
    rpr_scene scene_;
    rpr_framebuffer color_framebuffer_;
    rpr_camera rprCamera;
    std::uint32_t semaphore_index_;

    std::int32_t quality = 0;

    VkPhysicalDeviceDescriptorIndexingFeaturesEXT desc_indexing;

    struct bbox
    {
        RadeonProRender::float3 pmin;
        RadeonProRender::float3 pmax;

        RadeonProRender::float3 center() const
        {
            return (pmin + pmax) * 0.5f;
        }
    };

    struct GismoInfo // Gizmo related stuff.
    {
        static constexpr std::int32_t kNoObjectIndex = 0;

        std::int32_t selected_object_index = kNoObjectIndex;
        std::vector<std::string> object_names = { "" };

        bool is_local = false;

        std::vector<rpr_shape> shapes;
        std::vector<bbox> shape_aabbs;

        struct // Pipeline.
        {
            VkPipeline pipeline;
            VkPipelineLayout pipeline_layout;
            VkDescriptorSetLayout descriptor_set_layout;
            VkDescriptorSet descriptor_set;
        };

        struct 
        {
            glm::mat4 model;
            glm::mat4 view;
            glm::mat4 projection;
            std::uint32_t is_visible = 0;
            float scale = 1.0f;
        } ubo;

        vks::Buffer ubo_buffer;
        bool ubo_need_update = false;

        bool active() const 
        { 
            return selected_object_index != kNoObjectIndex &&
                (selected_object_index - 1) < shapes.size();
        }

    } gizmo;

    VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
    {
        zoom = -2.5f;
        rotation = { 0.0f, 15.0f, 0.0f };
        title = "RPR Gizmo";
        settings.overlay = true;
        enabledDeviceExtensions.push_back(VK_EXT_SHADER_SUBGROUP_BALLOT_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);

        enabledFeatures.shaderInt64 = VK_TRUE;
        enabledFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;
        enabledFeatures.fragmentStoresAndAtomics = VK_TRUE;
        enabledFeatures.geometryShader = VK_TRUE;
        enabledFeatures.independentBlend = VK_TRUE;
        enabledFeatures.wideLines = VK_TRUE;

        memset(&desc_indexing, 0, sizeof(desc_indexing));
        desc_indexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
        desc_indexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

        deviceCreatepNextChain = &desc_indexing;
    }

    ~VulkanExample()
    {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        rprObjectDelete(scene_);
        rprObjectDelete(color_framebuffer_);
        rprObjectDelete(mat_system_);
        rprObjectDelete(context_);

        for (auto i = 0u; i < frames_in_flight_; ++i)
        {
            vkDestroySemaphore(device, framebuffer_release_semaphores_[i], nullptr);
        }

        destroyTextureImage(texture);

        vkDestroyPipeline(device, pipelines.solid, nullptr);

        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vertexBuffer.destroy();
        indexBuffer.destroy();

        {
            vkDestroyPipeline(device, gizmo.pipeline, nullptr);
            vkDestroyPipelineLayout(device, gizmo.pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(device, gizmo.descriptor_set_layout, nullptr);
            gizmo.ubo_buffer.destroy();
        }
    }

    // Enable physical device features required for this example
    virtual void getEnabledFeatures()
    {
        // Enable anisotropic filtering if supported
        if (deviceFeatures.samplerAnisotropy) {
            enabledFeatures.samplerAnisotropy = VK_TRUE;
        };
    }

    // Free all Vulkan resources used by a texture object
    void destroyTextureImage(Texture texture)
    {
        vkDestroyImageView(device, texture.view, nullptr);
        //vkDestroyImage(device, texture.image, nullptr);
        vkDestroySampler(device, texture.sampler, nullptr);
        //vkFreeMemory(device, texture.deviceMemory, nullptr);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers()
    {
        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &gizmo.ubo_buffer,
            sizeof(gizmo.ubo)));

        // Map persistent
        VK_CHECK_RESULT(gizmo.ubo_buffer.map());

        updateUniformBuffers();
    }

    glm::mat4 makeInfinitePerspectiveMatrix(float fovy, float aspect, float n)
    {
        glm::mat4 m;

        const float t = std::tan(fovy);
        
        m[0][0] = 1.0f / (t * aspect);
        m[1][1] = -1.0f / t;

        m[2][2] = 0.0f;
        m[2][3] = n;

        m[3][2] = -1.0f;
        m[3][3] = 0.0f;

        return m;
    }

    void updateUniformBuffers()
    {
        gizmo.ubo.projection = cameraController.getProjection();
        gizmo.ubo.view = cameraController.getView();

        if (gizmo.active())
        {
            rpr_shape shape = gizmo.shapes[gizmo.selected_object_index - 1];
            glm::mat4 shape_transform;
            CHECK_RPR(rprShapeGetInfo(shape, RPR_SHAPE_TRANSFORM,
                sizeof(glm::mat4), &shape_transform, nullptr));

            auto center = gizmo.shape_aabbs[gizmo.selected_object_index - 1].center();
            auto center_translate = glm::translate({}, glm::vec3(center.x, center.y, center.z));

            auto gizmo_model = shape_transform * center_translate;

            if (!gizmo.is_local)
            {
                glm::mat4 translate_only;
                translate_only[3][0] = gizmo_model[3][0];
                translate_only[3][1] = gizmo_model[3][1];
                translate_only[3][2] = gizmo_model[3][2];

                gizmo_model = translate_only;
            }
    
            gizmo.ubo.model = gizmo_model;

            // Compute gizmo scale.
            // Scale gizmo corner vertices to get the gizmo screen size to be independent on depth.
            glm::vec4 zero_projected = gizmo.ubo.projection * gizmo.ubo.view * gizmo.ubo.model * 
                glm::vec4(glm::vec3(0.0), 1.0);
            float depth = zero_projected.z / zero_projected.w;
            const float kGizmoScaleFactor = 0.01;
            gizmo.ubo.scale = kGizmoScaleFactor / depth;
        }

        std::memcpy(gizmo.ubo_buffer.mapped, &gizmo.ubo, sizeof(gizmo.ubo));
    }

    void buildCommandBuffers()
    {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        VkClearValue clearValues[2];
        clearValues[0].color = defaultClearColor;
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent.width = width;
        renderPassBeginInfo.renderArea.extent.height = height;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
        {
            // Set target frame buffer
            renderPassBeginInfo.framebuffer = frameBuffers[i];

            VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.solid);

            //VkDeviceSize offsets[1] = { 0 };

            vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

            // Call gizmo pipeline.

            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, gizmo.pipeline);
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, gizmo.pipeline_layout,
                0, 1, &gizmo.descriptor_set, 0, nullptr);

            vkCmdSetLineWidth(drawCmdBuffers[i], 3.0f);
            vkCmdDraw(drawCmdBuffers[i], 6, 1, 0, 0);

            drawUI(drawCmdBuffers[i]);

            vkCmdEndRenderPass(drawCmdBuffers[i]);

            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
        }
    }

    void draw()
    {
        VulkanExampleBase::prepareFrame();

        std::array<VkSemaphore, 2> wait_semaphores =
        {
            semaphores.presentComplete,
            framebuffer_ready_semaphores_[semaphore_index_]
        };

        std::array<VkSemaphore, 2> signal_semaphores =
        {
            semaphores.renderComplete,
            framebuffer_release_semaphores_[semaphore_index_]
        };
        std::array<VkPipelineStageFlags, 2> wait_stages = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

        submitInfo.pWaitDstStageMask = wait_stages.data();

        // Command buffer to be sumitted to the queue
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

        submitInfo.waitSemaphoreCount = wait_semaphores.size();
        submitInfo.pWaitSemaphores = wait_semaphores.data();

        submitInfo.signalSemaphoreCount = signal_semaphores.size();
        submitInfo.pSignalSemaphores = signal_semaphores.data();

        // Submit to queue
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

        VulkanExampleBase::submitFrame();
    }

    void setupVertexDescriptions()
    {
        // Binding description
        vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
        vertices.inputState.vertexBindingDescriptionCount = 0;
        vertices.inputState.pVertexBindingDescriptions = nullptr;
        vertices.inputState.vertexAttributeDescriptionCount = 0;
        vertices.inputState.pVertexAttributeDescriptions = nullptr;
    }

    void setupDescriptorPool()
    {
        // Example uses one ubo and one image sampler
        std::vector<VkDescriptorPoolSize> poolSizes =
        {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
        };

        VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            static_cast<uint32_t>(poolSizes.size()),
            poolSizes.data(), 2);

        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
    }

    void setupDescriptorSetLayout()
    {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
        {
            // Binding 0 : Fragment shader image sampler
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_SHADER_STAGE_FRAGMENT_BIT,
                0)
        };

        VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(
            setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

        VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &descriptorSetLayout,
            1);

        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));
    }

    void setupDescriptorSet()
    {
        VkDescriptorSetAllocateInfo allocInfo =
        vks::initializers::descriptorSetAllocateInfo(
            descriptorPool,
            &descriptorSetLayout,
            1);

        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

        // Setup a descriptor image info for the current texture to be used as a combined image sampler
        VkDescriptorImageInfo textureDescriptor;
        textureDescriptor.imageView = texture.view;				// The image's view (images are never directly accessed by the shader, but rather through views defining subresources)
        textureDescriptor.sampler = texture.sampler;			// The sampler (Telling the pipeline how to sample the texture, including repeat, border, etc.)
        textureDescriptor.imageLayout = texture.imageLayout;	// The current layout of the image (Note: Should always fit the actual use, e.g. shader read)

        std::vector<VkWriteDescriptorSet> writeDescriptorSets =
        {
                // Binding 1 : Fragment shader texture sampler
                //	Fragment shader: layout (binding = 1) uniform sampler2D samplerColor;
                vks::initializers::writeDescriptorSet(
                    descriptorSet,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,		// The descriptor set will use a combined image sampler (sampler and image could be split)
                0,												// Shader binding point 1
                &textureDescriptor)								// Pointer to the descriptor image for our texture
        };

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
    }

    void preparePipelines()
    {
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            0,
            VK_FALSE);

        VkPipelineRasterizationStateCreateInfo rasterizationState =
        vks::initializers::pipelineRasterizationStateCreateInfo(
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            VK_FRONT_FACE_COUNTER_CLOCKWISE,
            0);

        VkPipelineColorBlendAttachmentState blendAttachmentState =
        vks::initializers::pipelineColorBlendAttachmentState(
            0xf,
            VK_FALSE);

        VkPipelineColorBlendStateCreateInfo colorBlendState =
        vks::initializers::pipelineColorBlendStateCreateInfo(
            1,
            &blendAttachmentState);

        VkPipelineDepthStencilStateCreateInfo depthStencilState =
        vks::initializers::pipelineDepthStencilStateCreateInfo(
            VK_TRUE,
            VK_TRUE,
            VK_COMPARE_OP_LESS_OR_EQUAL);

        VkPipelineViewportStateCreateInfo viewportState =
        vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

        VkPipelineMultisampleStateCreateInfo multisampleState =
        vks::initializers::pipelineMultisampleStateCreateInfo(
            VK_SAMPLE_COUNT_1_BIT,
            0);

        std::vector<VkDynamicState> dynamicStateEnables = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState =
        vks::initializers::pipelineDynamicStateCreateInfo(
            dynamicStateEnables.data(),
            static_cast<uint32_t>(dynamicStateEnables.size()),
            0);

        // Load shaders
        std::array<VkPipelineShaderStageCreateInfo,2> shaderStages;

        shaderStages[0] = loadShader(getAssetPath() + "shaders/rpr_base_render/fullscreen_quad.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/rpr_base_render/output.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        VkGraphicsPipelineCreateInfo pipelineCreateInfo =
        vks::initializers::pipelineCreateInfo(
            pipelineLayout,
            renderPass,
            0);

        pipelineCreateInfo.pVertexInputState = &vertices.inputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();

        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.solid));
    }

    void initRpr()
    {
        //Register plugin
        rpr_int plugin_id = rprRegisterPlugin(PLUGIN_NAME);

        //Initialize rpr context with VK interop
        VkInteropInfo interop_info;
        VkInteropInfo::VkInstance instance;
        instance.device = device;
        instance.physical_device = physicalDevice;
        interop_info.instances = &instance;
        interop_info.instance_count = 1;
        interop_info.main_instance_index = 0;
        interop_info.frames_in_flight = frames_in_flight_;

        //Create framebuffer release semaphores required by RPR
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (std::uint32_t a = 0; a < frames_in_flight_; ++a)
        {
            VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &framebuffer_release_semaphores_[a]));
        }
        interop_info.framebuffers_release_semaphores = framebuffer_release_semaphores_.data();

        // Set release semaphores to signalled state
        VkCommandBuffer fake_cmd_buffer;
        fake_cmd_buffer = createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VK_CHECK_RESULT(vkEndCommandBuffer(fake_cmd_buffer));

        VkSubmitInfo info;
        memset(&info, 0, sizeof(VkSubmitInfo));
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.pSignalSemaphores = framebuffer_release_semaphores_.data();
        info.signalSemaphoreCount = framebuffer_release_semaphores_.size();
        info.pWaitSemaphores = nullptr;
        info.waitSemaphoreCount = 0;
        info.pCommandBuffers = &fake_cmd_buffer;
        info.commandBufferCount = 1;

        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE));
        VK_CHECK_RESULT(vkQueueWaitIdle(queue));

        rpr_context_properties properties[] =
        {
            (void*)RPR_CONTEXT_CREATEPROP_VK_INTEROP_INFO, &interop_info,
            (void*)RPR_CONTEXT_CREATEPROP_HYBRID_ACC_MEMORY_SIZE, &acc_size_,
            (void*)RPR_CONTEXT_CREATEPROP_HYBRID_VERTEX_MEMORY_SIZE, &vbuf_size_,
            (void*)RPR_CONTEXT_CREATEPROP_HYBRID_INDEX_MEMORY_SIZE, &ibuf_size_,
            (void*)RPR_CONTEXT_CREATEPROP_HYBRID_STAGING_MEMORY_SIZE, &sbuf_size_,
            0
        };

        CHECK_RPR(rprCreateContext(RPR_API_VERSION, &plugin_id, 1,
            RPR_CREATION_FLAGS_ENABLE_GPU0 | RPR_CREATION_FLAGS_ENABLE_VK_INTEROP,
            properties, "cache", &context_));

        CHECK_RPR(rprContextSetParameterByKey1u(context_, RPR_CONTEXT_Y_FLIP, RPR_TRUE));

        //Get extension functions
        CHECK_RPR(rprContextGetFunctionPtr(context_, RPR_CONTEXT_FLUSH_FRAMEBUFFERS_FUNC_NAME, (void**)(&rprContextFlushFrameBuffers)));

        //Create material system
        CHECK_RPR(rprContextCreateMaterialSystem(context_, 0, &mat_system_));

        //Create scene
        CHECK_RPR(rprContextCreateScene(context_, &scene_))
    }

    void initAovs()
    {
        rpr_framebuffer_desc desc = { (rpr_uint)width, (rpr_uint)height};
        rpr_framebuffer_format fmt = { 4, RPR_COMPONENT_TYPE_FLOAT16 };

        CHECK_RPR(rprContextCreateFrameBuffer(context_, fmt, &desc, &color_framebuffer_));
        //Get semaphores from color aov
        CHECK_RPR(rprContextGetInfo(context_, RPR_CONTEXT_FRAMEBUFFERS_READY_SEMAPHORES,
             sizeof(VkSemaphore) * frames_in_flight_, framebuffer_ready_semaphores_.data(), nullptr));

        CHECK_RPR(rprContextSetAOV(context_, RPR_AOV_COLOR, color_framebuffer_));
    }

    void initScene()
    {
        // Load up basic scene
        CHECK_RPR(rprLoadScene("../data/models/CornellBox/orig.objm", 
            "../data/models/CornellBox/", context_, mat_system_, &scene_));
        CHECK_RPR(rprContextSetScene(context_, scene_));

        //Init camera
        CHECK_RPR(rprContextCreateCamera(context_, &rprCamera));
        CHECK_RPR(rprCameraSetMode(rprCamera, RPR_CAMERA_MODE_PERSPECTIVE));

        glm::vec3 eye = glm::vec3(0.0f, 1.5f, 10.0f);
        glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);
        glm::vec3 at = glm::vec3(0.0f, 1.5f, 0.0f);

        glm::vec2 sensor_size(0.036f, 0.024f);
        float focal_length = 0.035f;

        const float fovy = atan(sensor_size.y / (2.0f * focal_length));
        const float aspect = sensor_size.x / sensor_size.y;

        cameraController.setPerspective(fovy, aspect, 0.1f, 10000.f);
        cameraController.LookAt(eye, at, up);
        
        CHECK_RPR(rprCameraSetSensorSize(rprCamera, sensor_size.x * 1000.f, sensor_size.y * 1000.f)); //Standart 36x24 sensor
        CHECK_RPR(rprSceneSetCamera(scene_, rprCamera));

        rpr_light env_light = nullptr;
        CHECK_RPR(rprContextCreateEnvironmentLight(context_, &env_light));
        rpr_image image = nullptr;
        CHECK_RPR(rprContextCreateImageFromFile(context_, "../data/textures/hdr/studio015.hdr", &image));
        CHECK_RPR(rprEnvironmentLightSetImage(env_light, image));
        CHECK_RPR(rprSceneSetEnvironmentLight(scene_, env_light));

        CHECK_RPR(rprContextSetParameterByKey1f(context_, RPR_CONTEXT_DISPLAY_GAMMA, 2.2f));

        // Get object list.

        std::size_t shapes_count = 0;
        CHECK_RPR(rprSceneGetInfo(scene_, RPR_SCENE_SHAPE_COUNT, 0, nullptr, &shapes_count));

        gizmo.shapes.resize(shapes_count);
        CHECK_RPR(rprSceneGetInfo(scene_, RPR_SCENE_SHAPE_LIST, shapes_count * sizeof(rpr_shape), 
            gizmo.shapes.data(), nullptr));

        for (auto shape_index = 0; shape_index < gizmo.shapes.size(); ++shape_index)
        {
            auto shape = gizmo.shapes[shape_index];

            std::size_t name_size = 0;
            CHECK_RPR(rprShapeGetInfo(shape, RPR_SHAPE_NAME, 0, nullptr, &name_size));
            std::vector<char> name(name_size);
            CHECK_RPR(rprShapeGetInfo(shape, RPR_SHAPE_NAME, name_size, name.data(), nullptr));

            if (name.size() > 1)
            {
                gizmo.object_names.push_back(name.data());
            }
            else
            {
                gizmo.object_names.push_back("Shape " + std::to_string(shape_index));
            }

            // Get aabb of the shape.
            bbox shape_aabb;
            CHECK_RPR(rprMeshGetInfo(shape, RPR_MESH_AABB, sizeof(shape_aabb), &shape_aabb, nullptr));
            gizmo.shape_aabbs.push_back(shape_aabb);
        }
    }

    VkImage getRenderedImage()
    {
        VkImage image;
        auto status = rprFrameBufferGetInfo(color_framebuffer_, RPR_VK_IMAGE_OBJECT, sizeof(VkImage), &image, 0);

        return status == RPR_SUCCESS ? image : nullptr;
    }

    void loadTextureFromRprFb()
    {
        //get VkImage from FB
        VkImage image = getRenderedImage();

        // Create a texture sampler
        // In Vulkan textures are accessed by samplers
        // This separates all the sampling information from the texture data. This means you could have multiple sampler objects for the same texture with different settings
        // Note: Similar to the samplers available with OpenGL 3.3
        VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
        sampler.magFilter = VK_FILTER_NEAREST;
        sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.mipLodBias = 0.0f;
        sampler.compareOp = VK_COMPARE_OP_NEVER;
        sampler.minLod = 0.0f;
        // Set max level-of-detail to mip level count of the texture
        sampler.maxLod = 0.0f;
        // Enable anisotropic filtering
        // This feature is optional, so we must check if it's supported on the device
        sampler.maxAnisotropy = 1.0;
        sampler.anisotropyEnable = VK_FALSE;
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &texture.sampler));

        // Create image view
        // Textures are not directly accessed by the shaders and
        // are abstracted by image views containing additional
        // information and sub resource ranges
        VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
        // The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
        // It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.baseMipLevel = 0;
        view.subresourceRange.baseArrayLayer = 0;
        view.subresourceRange.layerCount = 1;
        // Linear tiling usually won't support mip maps
        // Only set mip map count if optimal tiling is used
        view.subresourceRange.levelCount = 1;
        // The view will be based on the texture's image
        view.image = image;
        VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &texture.view));

        texture.image = image;

        texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    }
    
    void setupGizmoPipeline()
    {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
        {
            // Binding 0 : Fragment shader image sampler
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                VK_SHADER_STAGE_VERTEX_BIT,
                0)
        };

        VkDescriptorSetLayoutCreateInfo descriptorLayout =
            vks::initializers::descriptorSetLayoutCreateInfo(
                setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &gizmo.descriptor_set_layout));

        VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
            vks::initializers::pipelineLayoutCreateInfo(
                &gizmo.descriptor_set_layout,
                1);

        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &gizmo.pipeline_layout));

        auto allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool,
            &gizmo.descriptor_set_layout, 1);

        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &gizmo.descriptor_set));

        std::vector<VkWriteDescriptorSet> writeDescriptorSets =
        {
            vks::initializers::writeDescriptorSet(
                gizmo.descriptor_set,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            0,
            &gizmo.ubo_buffer.descriptor)
        };

        vkUpdateDescriptorSets(device,
            static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
            vks::initializers::pipelineInputAssemblyStateCreateInfo(
                VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                0,
                VK_FALSE);

        VkPipelineRasterizationStateCreateInfo rasterizationState =
            vks::initializers::pipelineRasterizationStateCreateInfo(
                VK_POLYGON_MODE_FILL,
                VK_CULL_MODE_NONE,
                VK_FRONT_FACE_COUNTER_CLOCKWISE,
                0);

        VkPipelineColorBlendAttachmentState blendAttachmentState =
            vks::initializers::pipelineColorBlendAttachmentState(
                0xf,
                VK_FALSE);

        VkPipelineColorBlendStateCreateInfo colorBlendState =
            vks::initializers::pipelineColorBlendStateCreateInfo(
                1,
                &blendAttachmentState);

        VkPipelineDepthStencilStateCreateInfo depthStencilState =
            vks::initializers::pipelineDepthStencilStateCreateInfo(
                VK_FALSE,
                VK_FALSE,
                VK_COMPARE_OP_ALWAYS);

        VkPipelineViewportStateCreateInfo viewportState =
            vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

        VkPipelineMultisampleStateCreateInfo multisampleState =
            vks::initializers::pipelineMultisampleStateCreateInfo(
                VK_SAMPLE_COUNT_1_BIT,
                0);

        std::vector<VkDynamicState> dynamicStateEnables = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_LINE_WIDTH
        };
        VkPipelineDynamicStateCreateInfo dynamicState =
            vks::initializers::pipelineDynamicStateCreateInfo(
                dynamicStateEnables.data(),
                static_cast<uint32_t>(dynamicStateEnables.size()),
                0);

        // Load shaders
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        shaderStages[0] = loadShader(getAssetPath() + "shaders/rpr_gizmo/gizmo.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/rpr_gizmo/gizmo.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        VkGraphicsPipelineCreateInfo pipelineCreateInfo =
            vks::initializers::pipelineCreateInfo(
                gizmo.pipeline_layout,
                renderPass,
                0);

        pipelineCreateInfo.pVertexInputState = &vertices.inputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();

        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo,
            nullptr, &gizmo.pipeline));

    }

    void prepare()
    {
        VulkanExampleBase::prepare();

        initRpr();
        initAovs();
        initScene();

        loadTextureFromRprFb();

        prepareUniformBuffers();
        setupVertexDescriptions();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        setupGizmoPipeline();
        buildCommandBuffers();
        prepared = true;
    }

    virtual void render()
    {
        if (!prepared)
            return;

        CHECK_RPR(rprContextRender(context_));
        CHECK_RPR(rprContextFlushFrameBuffers(context_));

        CHECK_RPR(rprContextGetInfo(context_, RPR_CONTEXT_INTEROP_SEMAPHORE_INDEX,
            sizeof(semaphore_index_), &semaphore_index_, nullptr));

        if (gizmo.active())
        {
            updateUniformBuffers();
        }

        draw();

        if (gizmo.selected_object_index != gizmo.kNoObjectIndex)
        {
            rpr_shape shape = gizmo.shapes[gizmo.selected_object_index - 1];
            glm::mat4 shape_transform;
            CHECK_RPR(rprShapeGetInfo(shape, RPR_SHAPE_TRANSFORM,
                sizeof(glm::mat4), &shape_transform, nullptr));

            shape_transform = glm::translate(shape_transform, glm::vec3(0.005f, 0.0f, 0.0f));
            shape_transform = glm::rotate(shape_transform, 0.01f, { 1, 1, 1 });

            CHECK_RPR(rprShapeSetTransform(shape, false, 
                reinterpret_cast<const rpr_float*>(&shape_transform)));
        }
    }

    virtual void viewChanged()
    {
        auto pos = cameraController.camera->GetPosition();
        auto up = cameraController.camera->GetUpVector();
        auto at = cameraController.camera->GetAt();

        CHECK_RPR(rprCameraLookAt(rprCamera,
            pos.x, pos.y, pos.z, at.x, at.y, at.z, up.x, up.y, up.z));

        CHECK_RPR(rprFrameBufferClear(color_framebuffer_));
    }

    void updateQuality()
    {
        CHECK_RPR(rprContextSetParameterByKey1u(context_, RPR_CONTEXT_RENDER_QUALITY, quality));
        CHECK_RPR(rprFrameBufferClear(color_framebuffer_));
    }

    virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
    {
        if (overlay->header("Settings"))
        {
            if (overlay->comboBox("Quality", &quality, {"Low", "Medium", "High"}))
            {
                updateQuality();
            }
        }

        if (overlay->header("Editor"))
        {
            if (overlay->comboBox("Selected object", &gizmo.selected_object_index, 
                gizmo.object_names))
            {
                editObject();
            }

            overlay->checkBox("Local", &gizmo.is_local);
        }
    }

    void editObject()
    {
        if (!gizmo.active())
        {
            // Reset editing.
            gizmo.ubo.is_visible = 0;
            updateUniformBuffers();
        }
        else
        {
            assert(gizmo.selected_object_index > 0 && 
                gizmo.selected_object_index < gizmo.object_names.size());

            // Start edit object.

            gizmo.ubo.is_visible = 1;
            updateUniformBuffers();
        }
    }

    void mouseMoved(double x, double y, bool &handled) override
    {
        if (!mouseButtons.left || !gizmo.active())
        {
            return;
        }

        // Define whether any of the gizmo axis was hitted.
        
        glm::vec2 screen_pos = { float(x), float(y) };
        glm::vec2 click_pos_ndc = (screen_pos / glm::vec2(width, height)) * 2.0f - glm::vec2(1.0f);

        const glm::vec4 basis_verts[] = 
            { {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f} };

        auto origin_projected = gizmo.ubo.projection * gizmo.ubo.view * gizmo.ubo.model * 
            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        origin_projected /= origin_projected.w;

        float min_eps = FLT_MAX;

        auto axis_index = 0u;
        auto clicked_axis_index = ~0u;

        for (const auto& basis_vert : basis_verts)
        {
            auto basis_projected = gizmo.ubo.projection * gizmo.ubo.view * gizmo.ubo.model *
                gizmo.ubo.scale * basis_vert;
            basis_projected /= basis_projected.w;

            float eps = std::fabs((click_pos_ndc.x - origin_projected.x) / (basis_projected.x - origin_projected.x) -
                (click_pos_ndc.y - origin_projected.y) / (basis_projected.y - origin_projected.y));

            constexpr float kClickEps = 0.2f;
            if (eps < kClickEps && eps < min_eps)
            {
                min_eps = eps;
                clicked_axis_index = axis_index;
            }

            ++axis_index;
        }

        if (clicked_axis_index != ~0)
        {
            handled = true;

            const char labels[] = { 'X', 'Y', 'Z' };

            std::cout << "Clicked: " << labels[axis_index] << ", eps = " << min_eps << std::endl;
        }
        


    }
};

VULKAN_EXAMPLE_MAIN()
