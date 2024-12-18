// This file was generated by a tool; you should avoid making direct changes.
// Consider using 'partial classes' to extend these types
// Input: api_def.proto

#pragma warning disable CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class ApiDef : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"graph_op_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string GraphOpName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2)]
        public Visibility visibility { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"endpoint")]
        public global::System.Collections.Generic.List<Endpoint> Endpoints { get; } = new global::System.Collections.Generic.List<Endpoint>();

        [global::ProtoBuf.ProtoMember(4, Name = @"in_arg")]
        public global::System.Collections.Generic.List<Arg> InArgs { get; } = new global::System.Collections.Generic.List<Arg>();

        [global::ProtoBuf.ProtoMember(5, Name = @"out_arg")]
        public global::System.Collections.Generic.List<Arg> OutArgs { get; } = new global::System.Collections.Generic.List<Arg>();

        [global::ProtoBuf.ProtoMember(11, Name = @"arg_order")]
        public global::System.Collections.Generic.List<string> ArgOrders { get; } = new global::System.Collections.Generic.List<string>();

        [global::ProtoBuf.ProtoMember(6, Name = @"attr")]
        public global::System.Collections.Generic.List<Attr> Attrs { get; } = new global::System.Collections.Generic.List<Attr>();

        [global::ProtoBuf.ProtoMember(7, Name = @"summary")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Summary { get; set; } = "";

        [global::ProtoBuf.ProtoMember(8, Name = @"description")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Description { get; set; } = "";

        [global::ProtoBuf.ProtoMember(9, Name = @"description_prefix")]
        [global::System.ComponentModel.DefaultValue("")]
        public string DescriptionPrefix { get; set; } = "";

        [global::ProtoBuf.ProtoMember(10, Name = @"description_suffix")]
        [global::System.ComponentModel.DefaultValue("")]
        public string DescriptionSuffix { get; set; } = "";

        [global::ProtoBuf.ProtoContract()]
        public partial class Endpoint : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(1, Name = @"name")]
            [global::System.ComponentModel.DefaultValue("")]
            public string Name { get; set; } = "";

            [global::ProtoBuf.ProtoMember(2, Name = @"deprecation_message")]
            [global::System.ComponentModel.DefaultValue("")]
            public string DeprecationMessage { get; set; } = "";

        }

        [global::ProtoBuf.ProtoContract()]
        public partial class Arg : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(1, Name = @"name")]
            [global::System.ComponentModel.DefaultValue("")]
            public string Name { get; set; } = "";

            [global::ProtoBuf.ProtoMember(2, Name = @"rename_to")]
            [global::System.ComponentModel.DefaultValue("")]
            public string RenameTo { get; set; } = "";

            [global::ProtoBuf.ProtoMember(3, Name = @"description")]
            [global::System.ComponentModel.DefaultValue("")]
            public string Description { get; set; } = "";

        }

        [global::ProtoBuf.ProtoContract()]
        public partial class Attr : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(1, Name = @"name")]
            [global::System.ComponentModel.DefaultValue("")]
            public string Name { get; set; } = "";

            [global::ProtoBuf.ProtoMember(2, Name = @"rename_to")]
            [global::System.ComponentModel.DefaultValue("")]
            public string RenameTo { get; set; } = "";

            [global::ProtoBuf.ProtoMember(3, Name = @"default_value")]
            public AttrValue DefaultValue { get; set; }

            [global::ProtoBuf.ProtoMember(4, Name = @"description")]
            [global::System.ComponentModel.DefaultValue("")]
            public string Description { get; set; } = "";

        }

        [global::ProtoBuf.ProtoContract()]
        public enum Visibility
        {
            [global::ProtoBuf.ProtoEnum(Name = @"DEFAULT_VISIBILITY")]
            DefaultVisibility = 0,
            [global::ProtoBuf.ProtoEnum(Name = @"VISIBLE")]
            Visible = 1,
            [global::ProtoBuf.ProtoEnum(Name = @"SKIP")]
            Skip = 2,
            [global::ProtoBuf.ProtoEnum(Name = @"HIDDEN")]
            Hidden = 3,
        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class ApiDefs : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"op")]
        public global::System.Collections.Generic.List<ApiDef> Ops { get; } = new global::System.Collections.Generic.List<ApiDef>();

    }

}

#pragma warning restore CS0612, CS1591, CS3021, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
