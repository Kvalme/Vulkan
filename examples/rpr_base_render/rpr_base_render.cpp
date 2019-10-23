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

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION true

#ifndef WIN32
    #define PLUGIN_NAME "../libs/Rpr/Hybrid.so"
#else
    #define PLUGIN_NAME "../libs/Rpr/Hybrid.dll"
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


// Vertex layout for this example
struct Vertex {
    float pos[3];
    float uv[2];
    float normal[3];
};

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

    vks::Buffer uniformBufferVS;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 viewPos;
        float lodBias = 0.0f;
    } uboVS;

    struct {
        VkPipeline solid;
    } pipelines;

    VkPipelineLayout pipelineLayout;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    //Interop
    constexpr static std::uint32_t frames_in_flight_ = 3;
    //Default values for RPR. Can be omited
    std::uint32_t acc_size_ = 2048 * 1024u * 1024u;
    std::uint32_t vbuf_size_ = 1024 * 1024u * 1024u;
    std::uint32_t ibuf_size_ = 1024 * 1024u * 1024u;
    std::uint32_t sbuf_size_ = 512 * 1024u * 1024u;

    std::array<VkSemaphore, frames_in_flight_> framebuffer_release_semaphores_;
    std::array<VkSemaphore, frames_in_flight_> framebuffer_ready_semaphores_;

    rprContextFlushFrameBuffers_func rprContextFlushFrameBuffers;

    // RPR
    rpr_context context_;
    rpr_material_system mat_system_;
    rpr_scene scene_;
    rpr_framebuffer color_framebuffer_;
    rpr_camera camera_;


    VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
    {
        zoom = -2.5f;
        rotation = { 0.0f, 15.0f, 0.0f };
        title = "Texture loading";
        settings.overlay = true;
    }

    ~VulkanExample()
    {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        destroyTextureImage(texture);

        vkDestroyPipeline(device, pipelines.solid, nullptr);

        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vertexBuffer.destroy();
        indexBuffer.destroy();
        uniformBufferVS.destroy();
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
        vkDestroyImage(device, texture.image, nullptr);
        vkDestroySampler(device, texture.sampler, nullptr);
        vkFreeMemory(device, texture.deviceMemory, nullptr);
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

            drawUI(drawCmdBuffers[i]);

            vkCmdEndRenderPass(drawCmdBuffers[i]);

            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
        }
    }

    void draw()
    {
        VulkanExampleBase::prepareFrame();

        // Command buffer to be sumitted to the queue
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

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
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)
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

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers()
    {
        // Vertex shader uniform buffer block
        VK_CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &uniformBufferVS,
            sizeof(uboVS),
            &uboVS));

        updateUniformBuffers();
    }

    void updateUniformBuffers()
    {
        // Vertex shader
        uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.001f, 256.0f);
        glm::mat4 viewMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, zoom));

        uboVS.model = viewMatrix * glm::translate(glm::mat4(1.0f), cameraPos);
        uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

        uboVS.viewPos = glm::vec4(0.0f, 0.0f, -zoom, 0.0f);

        VK_CHECK_RESULT(uniformBufferVS.map());
        memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
        uniformBufferVS.unmap();
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
    }

    void initScene()
    {
        // Load up basic scene
        CHECK_RPR(rprLoadScene("../data/models/CornellBox/orig.objm", "../data/models/CornellBox/",
            context_, mat_system_, &scene_));
        CHECK_RPR(rprContextSetScene(context_, scene_));

        //Init camera
        CHECK_RPR(rprContextCreateCamera(context_, &camera_));
        CHECK_RPR(rprCameraSetMode(camera_, RPR_CAMERA_MODE_PERSPECTIVE));

        glm::vec3 camFront;
        camFront.x = -cos(glm::radians(rotation.x)) * sin(glm::radians(rotation.y));
        camFront.y = sin(glm::radians(rotation.x));
        camFront.z = cos(glm::radians(rotation.x)) * cos(glm::radians(rotation.y));
        camFront = glm::normalize(camFront);

        CHECK_RPR(rprCameraLookAt(camera_,
            cameraPos.x, cameraPos.y, cameraPos.z,
            camFront.x, camFront.y, camFront.z,
            0.0f, 1.0f, 0.0f));
        CHECK_RPR(rprCameraSetSensorSize(camera_, 36.f, 24.f)); //Standart 36x24 sensor
        CHECK_RPR(rprSceneSetCamera(scene_, camera_));
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
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
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
        view.format = VK_FORMAT_R16G16B16A16_UNORM;
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

    }
    void prepare()
    {
        VulkanExampleBase::prepare();

        initRpr();
        initAovs();
        initScene();

        loadTextureFromRprFb();

        setupVertexDescriptions();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    virtual void render()
    {
        if (!prepared)
            return;
        draw();
    }

    virtual void viewChanged()
    {
        updateUniformBuffers();
    }

    virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
    {
        if (overlay->header("Settings")) {
            if (overlay->sliderFloat("LOD bias", &uboVS.lodBias, 0.0f, (float)texture.mipLevels)) {
                updateUniformBuffers();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
